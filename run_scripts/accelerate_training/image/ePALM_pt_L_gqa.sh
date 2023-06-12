
config=./configs/image/ePALM_pt_L_gqa.yaml
data_dir=data/vl_adapter/vlt5_dataset
output_dir=logs/epalm/ePALM_pt_L_gqa

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 --num_machines=1 accelerate_training/gqa.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best\
--text_model facebook/opt-6.7b \
--vision_model vit_large_patch16_224 \
--low_cpu
# --resume --checkpoint $WORK/logs/epalm/ePALM_pt_L_gqa/checkpoint_last.pth


# --evaluate --checkpoint $WORK/logs/epalm/ePALM_pt_L_gqa/checkpoint_best.pth
