NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun \
--nproc_per_node=$NUM_GPUS \
--master_port=12355 \
infer_datarater_multigpu_cogvlm.py \
--model_path ""\
--output_dir ""


