NUM_GPUS=$(nvidia-smi -L | wc -l)
# NUM_GPUS=1
echo "begin train"
torchrun --nproc_per_node=$NUM_GPUS --master_port=12355 train_rater.py \
--config=/m2v_intern/dingkaixin/alchemist/configs/LAION-30m/rater/config_d16_256_csv_30m_laion_dev_raterDepth8_proxyDepth30.json

echo "finish training"