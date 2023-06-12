# visopt_caption_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e3fixed_sharedcon_nomlp_opt_6_7b_64ep
config=./configs/image/ePALM_pt_L_caption.yaml
data_dir=data/vl_adapter/vlt5_dataset
output_dir=logs/epalm/ePALM_pt_L_caption

export LC_ALL=C 

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 --num_machines=1 accelerate_training/caption.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-6.7b \
--vision_model vit_large_patch16_224 \
--low_cpu
# --resume --checkpoint $WORK/logs/visopt/ePALM_pt_L_caption/checkpoint_last.pth

# --evaluate --checkpoint $WORK/logs/visopt/ePALM_pt_L_caption/checkpoint_best.pth

