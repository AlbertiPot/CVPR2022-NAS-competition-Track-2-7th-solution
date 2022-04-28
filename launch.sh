# train
CUDA_VISIBLE_DEVICES=4 python main.py \
--seed 2 \
--lr 0.002 \
--weight_decay 6e-4 \
--batch_size 32 \
--train_ratio 0.9 \
--save_name 'exp1'

# search
# CUDA_VISIBLE_DEVICES=1 python test.py \
# --seed 0 \
# --batch_size 32 \
# --train_ratio 0.9 \
# --save_name 'exp1'