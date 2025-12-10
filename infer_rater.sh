# rater预测数据分数
NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun \
--nproc_per_node=$NUM_GPUS \
--master_port=12355 \
infer_datarater_multigpu_cogvlm.py \
--model_path /m2v_intern/dingkaixin/alchemist/outdir/30m_rater_d4_d16_256-0828-multinodes-dev_raterDepth8_proxyDepth30_cogvlm/ar-ckpt-ep1.pth \
--output_dir /m2v_intern/precomputed_latents/250606_laion5b_cogvlm_30M/rater_csv_1208/laion_30m_rater_d4_d16_256-0828-multinodes-dev_raterDepth8_proxyDepth30_cogvlm_ar-ckpt-ep1



# 从有序数据集筛选子集
# --portion 百分比 --gsampling 使用高斯采样策略（false为top-k采样）
# python infer_datarater_post_process_ranking_cogvlm_selectSubset.py --portion 20 --gsampling