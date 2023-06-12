# visopt_vqa_vlt5_stok_6clst_l19_31_reptok_nocache_ptl10_ptlr1e3fixed_nomlp_sharedcon_opt_6_7b_vitl_stdsplit_ep64

config=./configs/image/ePALM_pt_L_vqa.yaml
data_dir=data/vl_adapter/vlt5_dataset
output_dir=logs/epalm/ePALM_pt_L_vqa_acc

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 --num_machines=1 accelerate_training/vqa.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-6.7b --vision_model vit_large_patch16_224 \
--use_accelerate \
--low_cpu

# --resume --checkpoint $WORK/logs/epalm/ePALM_pt_L_vqa/checkpoint_last.pth 

# --checkpoint /data/mshukor/logs/epalm/ePALM_pt_L_vqa/checkpoint_last.pth --evaluate --open_ended_eval \





