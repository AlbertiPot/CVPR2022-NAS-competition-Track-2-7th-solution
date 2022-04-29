# train
CUDA_VISIBLE_DEVICES=4 python main.py \
--seed 2 \
--lr 0.001 \
--weight_decay 5e-4 \
--batch_size 25 \
--save_name 'enco_deco_lr1e3_wd5e4_bz25'

# search
# CUDA_VISIBLE_DEVICES=4 python test.py \
# --seed 0 \
# --save_name 'enco_deco'