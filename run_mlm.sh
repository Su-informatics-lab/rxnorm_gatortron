#!/bin/bash

# set the environment variable for CUDA devices
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=29500 \
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
  --max_seq_length 512 \
  --fp16 \
  --gradient_accumulation_steps 4 \
  --save_total_limit 5 \
  --report_to wandb \
  --ddp_find_unused_parameters False
