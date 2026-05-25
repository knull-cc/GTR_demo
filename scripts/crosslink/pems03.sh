#!/usr/bin/env bash

model_name=CrossLink

root_path_name=./dataset/PEMS
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

seq_len=96
crosslink_lags=1,4,12,16,64
crosslink_rank=16

for pred_len in 12 24 48 96
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
      --enc_in 358 \
      --cycle 288 \
      --use_revin 0 \
      --train_epochs 30 \
      --patience 5 \
      --crosslink_lags $crosslink_lags \
      --crosslink_rank $crosslink_rank \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done
