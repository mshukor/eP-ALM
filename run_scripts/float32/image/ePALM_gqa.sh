# visopt_gqa_vlt5_stok_6clst_l19_31_reptok_nocache_ptl10_ptlr1e5fixed_opt_2_7b
config=./configs/image/ePALM_gqa.yaml
data_dir=data/vl_adapter/vlt5_dataset
output_dir=logs/epalm/ePALM_gqa

torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12303  float32/gqa.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \

# --evaluate --checkpoint $WORK/logs/epalm/ePALM_gqa/checkpoint_best.pth


# --resume --checkpoint $WORK/logs/epalm/ePALM_gqa/checkpoint_last.pth

