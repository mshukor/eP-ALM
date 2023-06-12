
# visopt_video_msvd_vqa_reptok_6clst_l19_31_nocache_noeostk_ptl10_ptlr1e5fixed_tformer_8f_bs8_longt_opt_2_7b
config=./configs/video/ePALM_video_qa_msvd.yaml
data_dir=data/MSVD
output_dir=logs/epalm/ePALM_video_qa_msvd

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 --num_machines=1 accelerate_training/video_vqa.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--text_model facebook/opt-2.7b \
--low_cpu
# --resume --checkpoint $WORK/logs/epalm/ePALM_video_qa_msvd/checkpoint_last.pth

# --checkpoint $WORK/logs/epalm/ePALM_video_qa_msvd/checkpoint_best.pth --evaluate 
