# visopt_video_msrvtt_vqa_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e5fixed_tformer_8f_bs8_longt_opt_2_7b
config=./configs/video/ePALM_video_qa_msrvtt.yaml
data_dir=data/MSRVTT
output_dir=logs/epalm/ePALM_video_qa_msrvtt

torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12325  video_vqa.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \

# --resume --checkpoint $WORK/logs/epalm/ePALM_video_qa_msrvtt/checkpoint_last.pth


