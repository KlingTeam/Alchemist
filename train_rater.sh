NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=2
echo "begin train"
# python run_utility.py --gpus $NUM_GPUS --size 50000 --interval 0.01
# sleep 604800
torchrun --nproc_per_node=$NUM_GPUS --master_port=12355 train_rater.py \
--config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_cogvlm_batchHead_total10000_test.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/rater/config_d16_256_csv_rater_precompute_dev_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_cogvlm_total3000_test.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/HPDv3/rater/config_d16_256_csv_rater_precompute_dev_HPDv3.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/pixelprose-3.5m/rater/config_d16_256_csv_rater_precompute_dev_pixelprose_3.5m.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_Flux-reason-6m_1101.json
# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_Flux-reason-6m_1031.json

# --config=/m2v_intern/zhouyang15/codes/STAR-T2I/configs/config_d16_256_csv_rater_precompute_dev_cogvlm_batchHead.json
# 
# --config=configs/config_d16_256_csv_rater_precompute_cogvlm_batchHead.json

# --config=configs/config_d16_256_csv_rater_precompute_dev_cogvlm.json
# --config=configs/config_d16_256_csv_rater_precompute.json

sleep 604800
echo "finish training"