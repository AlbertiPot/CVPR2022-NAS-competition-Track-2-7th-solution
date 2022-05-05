# train
CUDA_VISIBLE_DEVICES=5 python main.py \
--save_name 'duke_lr5e4_ed6_bsz32_r8_seed1_dp5_cos'

# search
# CUDA_VISIBLE_DEVICES=1 python test.py \
# --seed 2 \
# --batch_size 25 \
# --save_name 'enco_deco_lr2e3_wd6e4_bz25'