NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node=$NUM_GPUS \
--master_port=18632 train.py \
--config=config_d16_256_csv_30m_rank_20_cogvlm_density_topk.json