# train
CUDA_VISIBLE_DEVICES=0 python main.py \
--seed 0 \
--save_name 'ae_cplfwok_r8_bsz8_cos'

# search
# CUDA_VISIBLE_DEVICES=1 python test.py \
# --seed 2 \
# --batch_size 25 \
# --save_name 'enco_deco_lr2e3_wd6e4_bz25'