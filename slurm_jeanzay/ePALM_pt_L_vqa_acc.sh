#!/bin/bash
   
#SBATCH --job-name=ePALM_pt_L_vqa_acc
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/gpfswork/rech/dyf/ugz83ue/logs/slurm/ePALM_pt_L_vqa_acc_test.out
###SBATCH --nodelist=jean-zay-a101
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu-dev
###SBATCH -C v100-32g
###SBATCH -C a100 
#SBATCH -A gtw@v100
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr



# module load cpuarch/amd

cd ~/ep-alm
source ~/.bashrc

source activate epalm
rm  core-*

export TRANSFORMERS_CACHE=/gpfswork/rech/dyf/ugz83ue/.cache/huggingface/transformers

# torchrun --nproc_per_node=8 --master_addr="localhost" --master_port=12302  vqa_vlt5.py \

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 --num_machines=1 accelerate_training/vqa.py \
--config ./configs/image/ePALM_pt_L_vqa.yaml \
--output_dir $WORK/logs/visopt/ePALM_pt_L_vqa_acc \
--data_dir $SCRATCH/data/vl_adapter/vlt5_dataset  \
--save_best --text_model facebook/opt-6.7b --vision_model vit_large_patch16_224 \
--use_accelerate --low_cpu \
--checkpoint $WORK/logs/visopt/ePALM_pt_L_vqa_acc/checkpoint_last.pth --evaluate 

# --resume --checkpoint $WORK/logs/visopt/visopt_vqa_vlt5_stok_6clst_l19_31_reptok_nocache_ptl10_ptlr1e3fixed_nomlp_sharedcon_opt_6_7b_vitl_stdsplit_ep64/checkpoint_last.pth 



# --checkpoint /data/mshukor/logs/visopt/visopt_vqa/checkpoint_last.pth --evaluate --open_ended_eval \
# --eval_data_dir /data/mshukor/data/vl_adapter/vlt5_dataset





