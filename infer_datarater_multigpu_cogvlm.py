import argparse
import copy
import datetime
import os
import random
import time
from aiohttp import RequestInfo
from tqdm import tqdm
import pandas as pd
import glob

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image

from models import build_vae_var, build_varRater
from models.text_encoder import build_text
from dataset.data import build_dataset_csv_raterInfer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def init_worker(worker_id: int):
    Image.MAX_IMAGE_PIXELS = None

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    args.rank = dist.get_rank()
    
    args.data_args = {'train_csv_path': '', "cogvlm":True}
    args.precomputed_latent = False
    args.precomputed_rootdir = ""
    args.data_load_reso = 256

    torch.manual_seed(args.seed + args.rank) 
    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    dataset_train, dataset_val, dataset_test = build_dataset_csv_raterInfer(args)
    
    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=False
    )
    ld_train = DataLoader(
        dataset_train, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, sampler=train_sampler,
        worker_init_fn=init_worker
    )

    text_encoder, in_dim_cross = build_text(pretrained_path=args.text_model_path, device=device)

    vae_model, var_model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=False, attn_l2_norm=True,
        enable_cross=True,
        in_dim_cross=in_dim_cross,
        flash_if_available=False, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True, lvl_emb=True,
        enable_logit_norm=args.enable_logit_norm,
        enable_adaptive_norm=False,
        train_mode='none',
        rope_theta=10000,
        rope_norm=64.0,
        sample_from_idx=9
    )
    if hasattr(args, "score_head_gelu"):
        score_head_gelu = args.score_head_gelu
    else:
        score_head_gelu = False
    varRater_model = build_varRater(
        vae_local=vae_model, device=device, patch_nums=args.patch_nums,
        depth=args.raterDepth, shared_aln=False, attn_l2_norm=True,
        enable_cross=True, in_dim_cross=in_dim_cross,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1,
        rope_emb=True, lvl_emb=True,
        rope_norm=64,
        enable_logit_norm=args.enable_logit_norm,
        enable_adaptive_norm=False,
        train_mode='all',
        rope_theta=10000,
        vae_ada=False,
        score_head_gelu=score_head_gelu
    )

    vae_model.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=True)
    varRater_model.load_state_dict(torch.load(args.model_path, map_location='cpu')["trainer"]["varRater_wo_ddp"], strict=True)
    del var_model
    torch.cuda.empty_cache()

    varRater_model = DDP(varRater_model.to(device), device_ids=[args.local_rank])
    vae_model = vae_model.to(device)
    
    vae_model.eval()
    varRater_model.eval()
    text_encoder.eval()

    if args.rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dist.barrier()

    start_time = time.time()
    results = []
    for batch_idx, batch_data in enumerate(tqdm(ld_train, disable=args.rank != 0)):
        with torch.no_grad():
            inp = batch_data['image'].to(device)
            prompt_embeds = text_encoder.extract_text_features(batch_data['prompt'])
            ids = batch_data['id']
            B = inp.shape[0]

            encoder_hidden_states, attn_mask, pooled_embed = prompt_embeds
            gt_idx_Bl = vae_model.img_to_idxBl(inp)
            x_BLCv_wo_first_l = vae_model.quantize.idxBl_to_var_input(gt_idx_Bl) 

            ratingWeight = varRater_model(
                x_BLCv_wo_first_l = x_BLCv_wo_first_l,
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = attn_mask,
                encoder_pool_feat = pooled_embed
            ).squeeze(1)
            # print("in line 141 ratingWeight", ratingWeight.shape) #torch.tensor...
            ids = ids.squeeze()
            batch_results = np.column_stack([ids.numpy(), ratingWeight.cpu().numpy()])
            results.append(batch_results)
            
            if (batch_idx + 1) % 100 == 0:
                temp_df = pd.DataFrame(np.concatenate(results), columns=['id', 'rating'])
                temp_df.to_csv(f"{args.output_dir}/rank{args.rank}_batch_{batch_idx + 1:06d}.csv", index=False)
                results.clear()

    if results:
        temp_df = pd.DataFrame(np.concatenate(results), columns=['id', 'rating'])
        temp_df.to_csv(f"{args.output_dir}/rank{args.rank}_final_batch.csv", index=False)
    
    dist.barrier()
    
    if args.rank == 0:
        all_files = glob.glob(f"{args.output_dir}/rank*_batch_*.csv") + glob.glob(f"{args.output_dir}/rank*_final_batch.csv")
        dfs = []
        for f in all_files:
            dfs.append(pd.read_csv(f))
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f"{args.output_dir}/final_df.csv", index=False)
        
        # 清理临时文件
        for f in all_files:
            os.remove(f)
        
        end_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to rater model.",
        required=True
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The folder where the csv are stored",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size for generation.",
        default=256,
    )
    parser.add_argument(
        "--score_head_gelu",
        type=bool,
        help="using gelu in rater score head",
        default=False,
    )  
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model.",
        default="",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        help="The path to VAE model.",
        default="",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of data loading workers (default: 4)"
    )

    parser.add_argument(
        "--patch_nums",
        type=list,
        help="The patch numbers for the model.",
        default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="The depth of the model.",
        default=16,
    )
    parser.add_argument(
        "--raterDepth",
        type=int,
        help="The depth of the model.",
        default=8,
    )
    parser.add_argument(
        "--enable_logit_norm",
        type=bool,
        help="Enable logit normalization.",
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)

    parser.add_argument("--local_rank", type=int, default=0)  
    args = parser.parse_args()
    
    main(args)