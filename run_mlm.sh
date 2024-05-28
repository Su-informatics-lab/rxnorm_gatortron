#!/bin/bash

# set the environment variable for CUDA devices
export CUDA_VISIBLE_DEVICES=4,5,6,7

NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
NPROC_PER_NODE=4
# master address and port for communication (default values)
MASTER_ADDR=localhost
MASTER_PORT=12355

# Run the training script with torch.distributed.launch
python -m torch.distributed.launch \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  run_mlm.py \
  --model_name_or_path UFNLP/gatortron-base \
  --train_file rxnorm.txt \
  --validation_split_percentage 5 \
  --output_dir ./ckpts \
  --overwrite_output_dir \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 100 \
  --save_steps 100 \
  --save_total_limit 5 \
  --do_train \
  --do_eval \
  --logging_dir ./logs \
  --logging_steps 100 \
  --mlm_probability 0.15 \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --early_stopping_patience 20 \
  --fp16 \
  --gradient_accumulation_steps 1 \
  --save_total_limit 5 \
  --deepspeed "./ds_config.json"
