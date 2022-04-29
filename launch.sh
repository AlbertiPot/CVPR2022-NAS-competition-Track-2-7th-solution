# train
CUDA_VISIBLE_DEVICES=0 python main.py \
--seed 0 \
--lr 0.001 \
--weight_decay 6e-4 \
--batch_size 16 \
--save_name 'mix_aenopos'

# search
# CUDA_VISIBLE_DEVICES=1 python test.py \
# --seed 2 \
# --batch_size 25 \
# --save_name 'enco_deco_lr2e3_wd6e4_bz25'