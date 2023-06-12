# visopt_video_msrvtt_caption_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e5fixed_tformer_16f_longt_opt_2_7b
config=./configs/video/ePALM_video_caption_msrvtt.yaml
data_dir=data/MSRVTT
output_dir=logs/epalm/ePALM_video_caption_msrvtt


torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12325  video_caption.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \

# --resume --checkpoint $WORK/logs/epalm/ePALM_video_caption_msrvtt/checkpoint_last.pth

# --evaluate --checkpoint $WORK/logs/epalm/ePALM_video_caption_msrvtt/checkpoint_best.pth

