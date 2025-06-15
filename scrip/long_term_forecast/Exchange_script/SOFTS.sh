export CUDA_VISIBLE_DEVICES=0

model_name=SOFTS
trainepoch=200
tunmodel=1
cfintune=1
cseg_len=1
lr=0.00001
tran_py=run_fintune_early_stop.py

python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --patience 10 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch


python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --patience 10 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch


python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch \
  --patience 10 \
  --des 'Exp' \
  --itr 1


python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch \
  --patience 10 \
  --des 'Exp' \
  --itr 1
