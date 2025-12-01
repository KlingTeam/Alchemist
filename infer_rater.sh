# rater预测数据分数
NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun \
--nproc_per_node=$NUM_GPUS \
--master_port=12355 \
infer_datarater_multigpu_cogvlm.py \
--model_path /m2v_intern/zhouyang15/codes/STAR-T2I/outdir/HPD_2m_dev_rater/ar-ckpt-ep1.pth \
--output_dir /m2v_intern/precomputed_latents/250606_HPDv3_2M/HPDv3_2M_dev_rater_dev_1102_ep1



# 从有序数据集筛选子集
# --portion 百分比 --gsampling 使用高斯采样策略（false为top-k采样）
python infer_datarater_post_process_ranking_cogvlm_selectSubset.py --portion 20 --gsampling

python  /m2v_intern/zhouyang15/codes/STAR-T2I/infer_datarater_pose_process_ranking_cogvlm_1007.py --portion 20 --pruning_ratio 0.25

python  /m2v_intern/zhouyang15/codes/STAR-T2I/infer_datarater_pose_process_ranking_cogvlm_1007.py --portion 10 --pruning_ratio 0.25

python  /m2v_intern/zhouyang15/codes/STAR-T2I/infer_datarater_pose_process_ranking_cogvlm_1007.py --portion 50 --pruning_ratio 0.25


python  /m2v_intern/zhouyang15/codes/STAR-T2I/infer_datarater_pose_process_ranking_cogvlm_1007.py --portion 80 --pruning_ratio 0.15

python  /m2v_intern/zhouyang15/codes/STAR-T2I/infer_datarater_pose_process_ranking_cogvlm_1007.py --portion 90 --pruning_ratio 0.05