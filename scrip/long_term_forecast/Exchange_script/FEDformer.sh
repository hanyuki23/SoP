export CUDA_VISIBLE_DEVICES=0

model_name=FEDformer
trainepoch=300
tunmodel=1
lr=0.00001
cfintune=1
cseg_len=1
tran_py=run_fintune_early_stop2.py

<<<<<<< HEAD
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
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --learning_rate $lr \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --train_epochs $trainepoch

python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --learning_rate $lr \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --train_epochs $trainepoch

python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --learning_rate $lr \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --train_epochs $trainepoch
=======
# python -u $tran_py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --learning_rate $lr \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --train_epochs $trainepoch

# python -u $tran_py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --learning_rate $lr \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --train_epochs $trainepoch

# python -u $tran_py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --learning_rate $lr \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --train_epochs $trainepoch
>>>>>>> d162c8d4752e7083fd44d7c660f8f25bc5ce67f4

python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --channel_fintune $cfintune \
  --learning_rate $lr \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --train_epochs $trainepoch
