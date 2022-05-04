# train
CUDA_VISIBLE_DEVICES=1 python main.py \
--save_name 'cplfw_bsz8_dp4_r7_seed4'

# search
# CUDA_VISIBLE_DEVICES=1 python test.py \
# --seed 2 \
# --batch_size 25 \
# --save_name 'enco_deco_lr2e3_wd6e4_bz25'