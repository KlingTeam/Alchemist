# rater预测数据分数
NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun \
--nproc_per_node=$NUM_GPUS \
--master_port=12355 \
infer_datarater_multigpu_cogvlm.py \
--model_path /m2v_intern/dingkaixin/STAR-T2I/outdir/rater/30m_rater_d4_d16_256-0828-multinodes-dev_batchHead_raterDepth8_cogvlm_ep/ar-ckpt-ep3.pth \
--output_dir /m2v_intern/precomputed_latents/251127_laion_30m_degraded/laion_30m_degraded



# 从有序数据集筛选子集
# --portion 百分比 --gsampling 使用高斯采样策略（false为top-k采样）
# python infer_datarater_post_process_ranking_cogvlm_selectSubset.py --portion 20 --gsampling