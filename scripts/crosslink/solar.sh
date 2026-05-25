#!/usr/bin/env bash

model_name=CrossLink

root_path_name=./dataset/Solar
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar

seq_len=96
crosslink_lags=1,4,16,64,72
crosslink_rank=16

for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    python -u run_crosslink.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --cycle 144 \
      --use_revin 0 \
      --train_epochs 30 \
      --patience 5 \
      --crosslink_lags $crosslink_lags \
      --crosslink_rank $crosslink_rank \
      --itr 1 --batch_size 64 --learning_rate 0.003 --random_seed $random_seed
done
done
