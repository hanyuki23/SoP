export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
trainepoch=300
tunmodel=0
lr=0.0001
cfintune=0
cseg_len=1
tran_py=run_fintune_early_stop.py

python -u $tran_py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchsange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch

<<<<<<< HEAD
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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch

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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --channel_fintune $cfintune \
  --cseg_len $cseg_len \
  --tun_model $tunmodel \
  --learning_rate $lr \
  --train_epochs $trainepoch
=======
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
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --learning_rate $lr \
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
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --learning_rate $lr \
#   --train_epochs $trainepoch

# python -u $tran_py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --itr 1 \
#   --channel_fintune $cfintune \
#   --cseg_len $cseg_len \
#   --tun_model $tunmodel \
#   --learning_rate $lr \
#   --train_epochs $trainepoch
>>>>>>> d162c8d4752e7083fd44d7c660f8f25bc5ce67f4
