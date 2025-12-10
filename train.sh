NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node=$NUM_GPUS \
--master_port=18632 train.py \
--config=/m2v_intern/dingkaixin/alchemist/configs/LAION-30m/config_d16_256_csv_30m_fulltrain_laion_cogvlm_prune_gaussian_rank50_descend.json