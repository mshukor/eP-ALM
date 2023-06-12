# visopt_caption_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e5fixed_opt_2_7b

config=./configs/image/ePALM_caption.yaml
data_dir=data/vl_adapter/vlt5_dataset
output_dir=logs/epalm/ePALM_caption


torchrun --nproc_per_node=1 --master_addr="localhost" --master_port=12326  caption.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \

# --evaluate --checkpoint $WORK/logs/epalm/ePALM_caption/checkpoint_best.pth
# --resume --checkpoint $WORK/logs/epalm/ePALM_caption/checkpoint_last.pth


