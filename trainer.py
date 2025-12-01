import time
from typing import List, Optional, Tuple, Union
import os
import pdb

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.sample_subset import prob_subset_selection
from torch.utils.data import DataLoader
import numpy as np
import os.path as osp
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import random

import dist
from models import VAR, VQVAE, VectorQuantizer2, VARRater
from utils.amp_sc import AmpOptimizer, GradientCollector 
from utils.misc import MetricLogger, TensorboardLogger
from math import ceil
from utils.utils import format_sentence
from utils.mask_utils import Scheduler
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        ##TODO
        self.gradient_collector=GradientCollector()
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))#[[0,1],[1,5],[5,...]]
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.mask_scheduler=Scheduler()
    

    @torch.no_grad()
    def inference_pic(self, args, text_enc, cur_ep, cur_iter,top_k=600,top_p=0.8,w_mask=False):
        seed = 4 #@param {type:"number"}
        # seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.var_wo_ddp.eval();self.vae_local.eval();text_enc.eval()
        for i in range(ceil(len(args.instance_prompt)/args.infer_bsz)):
            prompt_cur=args.instance_prompt[i*args.infer_bsz:min((i+1)*args.infer_bsz,len(args.instance_prompt))]
            B_=len(prompt_cur)
            prompt_cur=prompt_cur+[""]*B_
            label = torch.tensor([args.default_label]*B_).to(args.device, non_blocking=True) 
            with torch.inference_mode():
                prompt_embeds,prompt_attention_mask,pooled_embed = text_enc.extract_text_features(prompt_cur)
                recon_B3HW = self.var_wo_ddp.autoregressive_infer_cfg(B=B_, label_B=label, 
                                                                    encoder_hidden_states=prompt_embeds,
                                                                    encoder_attention_mask=prompt_attention_mask,
                                                                    encoder_pool_feat=pooled_embed,
                                                                    cfg=args.cfg, top_k=top_k, 
                                                                    top_p=top_p,g_seed=seed,
                                                                    w_mask=w_mask)
            save_path=osp.join(args.local_out_dir_path,'val_imgs','ep%d_iter%d'%(cur_ep,cur_iter))
            try:
                if not osp.exists(save_path):
                    os.makedirs(save_path)
            except:
                print('failed to makedir %s'%save_path)
            if osp.exists(save_path):
                for i,label in enumerate(prompt_cur[:B_]):
                    chw = (recon_B3HW[i].permute(1, 2, 0).cpu()*255.).numpy().astype(np.uint8)#(hwc)
                    PImage.fromarray(chw).save(osp.join(save_path,'%s.png'%(format_sentence(label))))

        self.var_wo_ddp.train()


    @torch.no_grad()
    def eval_ep(self, args, ld_val, text_enc):
        print(f"===========> in eval_ep")
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for _obj_idx, obj in enumerate(ld_val):
            print(f"===========> in eval_ep {_obj_idx} / {len(ld_val)}")
            inp_B3HW=obj['image'].to(dist.get_device(), non_blocking=True)
            if not args.precomputed_latent:
                obj['prompt_embeds']=text_enc.extract_text_features(obj['prompt'])
            encoder_hidden_states,encoder_attention_mask,pooled_embed=obj['prompt_embeds']
            B, V = inp_B3HW.shape[0], self.vae_local.vocab_size
            
            if args.precomputed_latent:
                gt_idx_Bl = self.vae_local.quantize.f_to_idxBl_or_fhat(inp_B3HW, to_fhat='idx')
            else:
                gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)

            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)#返回的是scale的插值结果

            self.var_wo_ddp.forward
            print(f"===========> in eval_ep before var forward")
            logits_BLV,_,drop_idxs = self.var(x_BLCv_wo_first_l=x_BLCv_wo_first_l,#输入的是fhat，预测的是残差
                                                    encoder_hidden_states=encoder_hidden_states,
                                                    encoder_attention_mask=encoder_attention_mask,
                                                    encoder_pool_feat=pooled_embed)
            
            if not drop_idxs==None:
                gt_BL=gt_BL[:,drop_idxs]
                last_l=self.var.module.drop_start
                L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
                L_tail += self.val_loss(logits_BLV.data[:, last_l:].reshape(-1, V), gt_BL[:, last_l:].reshape(-1)) * B
                acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
                acc_tail += (logits_BLV.data[:, last_l:].argmax(dim=-1) == gt_BL[:, last_l:]).float().mean() * 100

            else:
                L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
                L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
                acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
                acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot]) if not drop_idxs==None\
                else L_mean.new_tensor([L_mean.item(), 0, acc_mean.item(), 0, tot])
        
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prompt_embeds: None, prog_si: int, prog_wp_it: float,
        precomputed_latent: bool = False
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # label_B是imagenet的class，inp_B3HW是输入图像
        # progressive training
        encoder_hidden_states,attn_mask,pooled_embed=prompt_embeds
        self.var.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        if precomputed_latent:
            gt_idx_Bl: List[ITen] = self.vae_local.quantize.f_to_idxBl_or_fhat(inp_B3HW, to_fhat='idx')
        else:
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)#gt_idx_Bl:所有scale对应的离散token的真值

        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)#返回的是scale的插值结果，注意x_BLCv_wo_first_l没有第一个scale

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV,_,drop_idxs = self.var(x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attn_mask,
                                  encoder_pool_feat=pooled_embed)
            if not drop_idxs==None:
                gt_BL=gt_BL[:,drop_idxs]
                last_l=self.var.module.drop_start
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(1, -1)
            else:
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                if not drop_idxs==None:
                    lw=1/gt_BL.shape[-1]
                else:
                    lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
            # print("in line 212 loss ", loss) # in line 212 loss  tensor(8.3174, device='cuda:0', grad_fn=<MeanBackward0>)
            
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        # if grad_norm is not None:
        #     print("Gradient norm after clipping:", grad_norm)
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                if drop_idxs==None:
                    Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                    acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
                else:
                    Ltail = self.val_loss(logits_BLV.data[:, last_l:].reshape(-1, V), gt_BL[:, last_l:].reshape(-1)).item()
                    acc_tail = (pred_BL[:, last_l:] == gt_BL[:, last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
            # accm是平均所有尺度的loss，acc_tail是最后一个level的loss

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 200 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    

    def train_sampler_masked(self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prompt_embeds: None, prog_si: int, prog_wp_it: float, precomputed_latent: bool = False,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # label_B是imagenet的class，inp_B3HW是输入图像
        # progressive training
        encoder_hidden_states,attn_mask,pooled_embed=prompt_embeds
        self.var.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        if precomputed_latent:
            gt_idx_Bl = self.vae_local.quantize.f_to_idxBl_or_fhat(inp_B3HW, to_fhat='idx')
        else:
            gt_idx_Bl = self.vae_local.img_to_quant_embed(inp_B3HW)#gt_idx_Bl:所有scale对应的离散token的真值
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        #self.from_idx=9
        self.from_idx=self.var_wo_ddp.from_idx
        bg,_=self.begin_ends[self.from_idx];_,ed=self.begin_ends[-1]
        gt_BL=gt_BL[:,bg:ed]
        embed_BLCv=self.quantize_local.embedding(gt_BL)
        #暂时是按照目前的idx做一个embed算的，后期看有没有必要把fhat算出来
        # embed_BLCv=torch.cat(embed_BlCv, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)#返回的是scale的插值结果，注意x_BLCv_wo_first_l没有第一个scale

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward_sampler
            logits_sampler,mask = self.var.module.forward_sampler(x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attn_mask,
                                  encoder_pool_feat=pooled_embed,
                                  embed_Cvae=embed_BLCv)
            
            # 最后一个channel表示mask的channel
            logits_sampler_masked=torch.cat([logits_sampler*(1-mask[...,None]),#B,L,V
                                            (torch.ones_like(logits_sampler[...,-1])*10*mask).unsqueeze(-1)],#B,L,1
                                            dim=-1)
            gt_BL_masked=(gt_BL*(1-mask)+torch.ones_like(gt_BL)*V*mask)
            loss = self.train_loss(logits_sampler_masked.view(-1, V+1), gt_BL_masked.view(-1)).view(B, -1)
            lw = self.loss_weight[:,bg:ed]
            loss = loss.mul(lw).sum(dim=-1).mean()
            
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_sampler_masked.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_sampler_masked.data.view(-1, V+1), gt_BL_masked.view(-1)).item()
            acc_mean = (((pred_BL*(1-mask)) == gt_BL_masked).float().sum()/((1-mask).sum())).item() * 100#只算预测的，不算mask的
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = Lmean
                acc_tail = acc_mean
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 200 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_sampler_masked.data[:, bg:ed].reshape(-1, V+1), gt_BL_masked[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def eval_sampler():
        pass
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        missing=None;unexpected=None
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
        return missing,unexpected




class VARRaterTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        varRater_wo_ddp: VARRater, varRater: DDP,
        var_opt: AmpOptimizer, varRater_opt: AmpOptimizer,
        label_smooth: float,
        ul_weight: float, ul_weight_decay: float,
    ):
        super(VARRaterTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.varRater = varRater
        self.varRater_wo_ddp: VARRater = varRater_wo_ddp  # after torch.compile
        self.varRater_opt = varRater_opt
        del self.varRater_wo_ddp.rng
        self.varRater_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))#[[0,1],[1,5],[5,...]]
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.mask_scheduler=Scheduler()

        self.ul_weight = ul_weight
        self.ul_weight_decay = ul_weight_decay


    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prompt_embeds: None, prog_si: int, prog_wp_it: float, 
        new_inp_B3HW: FTen, new_label_B: Union[ITen, FTen], new_prompt_embeds: None, precomputed_latent: bool = False,
        rater_reg: bool = False, fixVar: bool = False
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # label_B是imagenet的class，inp_B3HW是输入图像
        # progressive training
        encoder_hidden_states,attn_mask,pooled_embed=prompt_embeds
        # <<< tensor of new_prompt_embeds
        new_encoder_hidden_states,new_attn_mask,new_pooled_embed=new_prompt_embeds
        # >>> tensor of new_prompt_embeds

        self.var.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        if precomputed_latent:
            gt_idx_Bl: List[ITen] = self.vae_local.quantize.f_to_idxBl_or_fhat(inp_B3HW, to_fhat='idx')
        else:        
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)#gt_idx_Bl:所有scale对应的离散token的真值

        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)#返回的是scale的插值结果，注意x_BLCv_wo_first_l没有第一个scale

        # <<< forward of new_inp_B3HW
        if precomputed_latent:
            new_gt_idx_Bl: List[ITen] = self.vae_local.quantize.f_to_idxBl_or_fhat(new_inp_B3HW, to_fhat='idx')
        else: 
            new_gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(new_inp_B3HW)
        new_gt_BL = torch.cat(new_gt_idx_Bl, dim=1)
        new_x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(new_gt_idx_Bl) 
        # >>> forward of new_inp_B3HW

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV,_,drop_idxs = self.var(x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attn_mask,
                                  encoder_pool_feat=pooled_embed)
            # <<< forward of new_inp_B3HW
            new_logits_BLV,_,new_drop_idxs = self.var(x_BLCv_wo_first_l=new_x_BLCv_wo_first_l,
                                          encoder_hidden_states=new_encoder_hidden_states,
                                          encoder_attention_mask=new_attn_mask,
                                          encoder_pool_feat=new_pooled_embed)
            # >>> forward of new_inp_B3HW

            if not drop_idxs==None:
                gt_BL=gt_BL[:,drop_idxs]
                last_l=self.var.module.drop_start
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(1, -1)
            else:
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) ## loss.shape = [B, L] # L is num token
            # print("in line 508 loss ", loss)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                if not drop_idxs==None:
                    lw=1/gt_BL.shape[-1]
                else:
                    lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

            # <<< loss of new_logits_BLV
            if not new_drop_idxs==None:
                new_gt_BL=new_gt_BL[:,new_drop_idxs]
                new_loss = self.train_loss(new_logits_BLV.view(-1, V), new_gt_BL.view(-1)).view(1, -1)
            else:
                new_loss = self.train_loss(new_logits_BLV.view(-1, V), new_gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert new_logits_BLV.shape[1] == new_gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                if not new_drop_idxs==None:
                    lw=1/new_gt_BL.shape[-1]
                else:
                    lw = self.loss_weight
            new_loss = new_loss.mul(lw)  
            # print("in line 537 new_loss ", new_loss)  
            
            with torch.no_grad():
                ratingWeight = self.varRater(x_BLCv_wo_first_l=new_x_BLCv_wo_first_l,
                                    encoder_hidden_states=new_encoder_hidden_states,
                                    encoder_attention_mask=new_attn_mask,
                                    encoder_pool_feat=new_pooled_embed)
                # softmax on ratingWeight
                ratingWeight = torch.softmax(ratingWeight, dim=0)
                ##TODO### 
                # 保证下限为0.001
                # ratingWeight = torch.clamp(ratingWeight, min=0.001)
                # # 如果需要重新归一化，使得总和为1
                # ratingWeight = ratingWeight / ratingWeight.sum()
                ###TODO####
                # print("in line 547 ratingWeight", ratingWeight)
                # ##TODO### 画loss和score的曲线图
                # loss_draw = new_loss.sum(dim=-1).mean()
                # ratingWeight_draw = ratingWeight.sum(dim=-1).mean()
                # ##设置batchsize=1 然后找出loss和score间关系 
                # print(f"loss_draw: {loss_draw:.4f}, ratingWeight_draw: {ratingWeight_draw:.4f}")

            weighted_new_loss = (new_loss * ratingWeight).sum(dim=-1).mean()
            # weighted_new_loss = loss_draw * ratingWeight_draw ##TODO
            # >>> loss of new_logits_BLV

        # backward
        total_loss = self.ul_weight * loss + (1 - self.ul_weight) * weighted_new_loss #ul_weight=0.97
        # print("in line 550 ul_weight, total_loss", self.ul_weight, total_loss)
        if fixVar:
            total_loss = 0 * total_loss
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)

        if stepping and hasattr(self, 'gradient_collector'):
            if 'label' in locals():  # 或者其他标签变量名
                self.gradient_collector.add_label(label.item())

        # <<< backward for datarater
        ratingWeight = self.varRater(x_BLCv_wo_first_l=new_x_BLCv_wo_first_l,
                            encoder_hidden_states=new_encoder_hidden_states,
                            encoder_attention_mask=new_attn_mask,
                            encoder_pool_feat=new_pooled_embed)
        # softmax on ratingWeight
        ratingWeight = torch.softmax(ratingWeight, dim=0)
        # print("in line 562 rating weight ", ratingWeight)
        rater_loss = (new_loss.detach() * ratingWeight).sum(dim=-1).mean()
        # print("in line 563 rater_loss ", new_loss)
        if rater_reg:
            eps = 1e-8
            reg_weight = 0.01
            entropy = - (ratingWeight * torch.log(ratingWeight + eps) +
                        (1 - ratingWeight) * torch.log(1 - ratingWeight + eps))
            loss_reg  = - torch.mean(entropy)
            rater_loss = rater_loss + reg_weight * loss_reg
        rater_grad_norm, rater_scale_log2 = self.varRater_opt.backward_clip_step(loss=rater_loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                if drop_idxs==None:
                    Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                    acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
                else:
                    Ltail = self.val_loss(logits_BLV.data[:, last_l:].reshape(-1, V), gt_BL[:, last_l:].reshape(-1)).item()
                    acc_tail = (pred_BL[:, last_l:] == gt_BL[:, last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
            # accm是平均所有尺度的loss，acc_tail是最后一个level的loss

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 200 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
                # log rater loss
                tb_lg.update(head='AR_iter_loss', rater_loss=rater_loss.item(), step=g_it)
                if rater_reg:
                    tb_lg.update(head='AR_iter_loss', loss_reg=loss_reg.item(), step=g_it)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    
    def train_sampler_masked(self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prompt_embeds: None, prog_si: int, prog_wp_it: float, precompute_latent: bool = False
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # label_B是imagenet的class，inp_B3HW是输入图像
        # progressive training
        encoder_hidden_states,attn_mask,pooled_embed=prompt_embeds
        self.var.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl = self.vae_local.img_to_quant_embed(inp_B3HW)#gt_idx_Bl:所有scale对应的离散token的真值
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        #self.from_idx=9
        self.from_idx=self.var_wo_ddp.from_idx
        bg,_=self.begin_ends[self.from_idx];_,ed=self.begin_ends[-1]
        gt_BL=gt_BL[:,bg:ed]
        embed_BLCv=self.quantize_local.embedding(gt_BL)
        #暂时是按照目前的idx做一个embed算的，后期看有没有必要把fhat算出来
        # embed_BLCv=torch.cat(embed_BlCv, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)#返回的是scale的插值结果，注意x_BLCv_wo_first_l没有第一个scale

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward_sampler
            logits_sampler,mask = self.var.module.forward_sampler(x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attn_mask,
                                  encoder_pool_feat=pooled_embed,
                                  embed_Cvae=embed_BLCv)
            
            # 最后一个channel表示mask的channel
            logits_sampler_masked=torch.cat([logits_sampler*(1-mask[...,None]),#B,L,V
                                            (torch.ones_like(logits_sampler[...,-1])*10*mask).unsqueeze(-1)],#B,L,1
                                            dim=-1)
            gt_BL_masked=(gt_BL*(1-mask)+torch.ones_like(gt_BL)*V*mask)
            loss = self.train_loss(logits_sampler_masked.view(-1, V+1), gt_BL_masked.view(-1)).view(B, -1)
            lw = self.loss_weight[:,bg:ed]
            loss = loss.mul(lw).sum(dim=-1).mean()
            
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_sampler_masked.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_sampler_masked.data.view(-1, V+1), gt_BL_masked.view(-1)).item()
            acc_mean = (((pred_BL*(1-mask)) == gt_BL_masked).float().sum()/((1-mask).sum())).item() * 100#只算预测的，不算mask的
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = Lmean
                acc_tail = acc_mean
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 200 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_sampler_masked.data[:, bg:ed].reshape(-1, V+1), gt_BL_masked[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def eval_sampler():
        pass
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt', 'varRater_wo_ddp', 'varRater_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        missing=None;unexpected=None
        for k in ('var_wo_ddp', 'vae_local', 'var_opt', 'varRater_wo_ddp', 'varRater_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
        return missing,unexpected