NUM_GPUS=$(nvidia-smi -L | wc -l)
# NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=6
/m2v_intern/caiminghong/anaconda3/envs/SVG/bin/torchrun --nproc_per_node=$NUM_GPUS \
--master_port=18632 train.py \
--config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_density_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/config_d16_256_csv_30m_fulltrain_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_random_50_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_dev1_mid1000_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_50_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s80_rank_20_cogvlm_v1.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_dev1_total3000_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_40_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/config_d16_256_csv_30m_rank20_pruned0.25_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_random_80_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_rank80_pruned0.15_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/config_d16_256_csv_30m_rank80_pruned0.15_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/config_d16_256_csv_30m_rank50_pruned0.25_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/config_d16_256_csv_30m_random_50_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_30_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s80_rank_20_cogvlm_v1.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_rank20_pruned0.25_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank40_pruned0.25.json
# torchrun --nproc_per_node=$NUM_GPUS \
# --master_port=12355 train.py \
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_rank30_pruned0.25_flux_reason.json


# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank40_pruned0.25.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_rank_20_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_rank_20_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_10_cogvlm_gSampling_depth24.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_10_cogvlm_gSampling_depth8.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_20_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_fulltrain_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_50_flux_reason.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/flux-reason-6m/config_d16_256_csv_30m_fulltrain_flux_reason_6m.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_20_cogvlm_depth24.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_density_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s0_cogvlm_rank20_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_50_cogvlm_depth24.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_freq_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_aesthtic_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_dev1_select1000_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_aesthtic_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25_batchHead_ep5.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_random_50_cogvlm_depth24.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_aesthtic_topk.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth24.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_freq_selectCurve.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_freq_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_dev1_select1000_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_clarity_selectCurve.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_aesthtic_selectCurve.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_clarity_selectCurve.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_clarity_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_dev1_select1000_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_clarity_topk.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_cogvlm_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth24.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth8.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth8.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth24.json


# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank10_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_depth8.json


# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead_reverse.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_gSampling.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json

python /m2v_intern/caiminghong/tools_init/check_gpu_v2.py
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_20_cogvlm_gSampling.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json


# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s35_rank_20_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_depth8.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s65_rank_20_cogvlm.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s80_rank_20_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_depth4.json


# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s80_rank_20_cogvlm.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_s80_rank_20_cogvlm.json
# --config=configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_30m_fulltrain_cogvlm.json

# --config=configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling.json


#最前面
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank20_pruned0.25_batchHead.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/cogvlm/config_d16_256_csv_30m_rank_50_cogvlm_gSampling_rank50_pruned0.25_batchHead.json
