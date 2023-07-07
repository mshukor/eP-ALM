
# visopt_audio_audiocaps_caption_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e5fixed_ast_bs8_longt_lessmask_opt_2_7b
config=./configs/audio/ePALM_audio_caption.yaml
data_dir=data/audiocaps
output_dir=logs/epalm/ePALM_audio_caption

torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12325  float32/audio_caption.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \

# --evaluate --checkpoint $WORK/logs/epalm/ePALM_audio_caption/checkpoint_best.pth


