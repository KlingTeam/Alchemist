NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "begin train"
torchrun --nproc_per_node=$NUM_GPUS --master_port=12355 train_rater.py \
--config=config_d16_256_csv_rater_precompute_dev_cogvlm_batchHead.json
sleep 604800
echo "finish training"