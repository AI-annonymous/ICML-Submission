#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/VIT/%j_bash_run_explainer.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/VIT/run_explainer_$CURRENT.out
echo "MIMIC_CXR"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000

# Training scripts BB
# Training from scratch
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_mimic_cxr.py \
#   --model_type='ViT-B_32_densenet' \
#   --learning_rate=0.03 \
#   --resize=512 \
#   --loss="CE" \
#   --selected-obs="pneumothorax" \
#   --pretrained="n" \
#   --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_mimic_cxr.py \
  --model_type='ViT-B_32' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumothorax" \
  --pretrained="y" \
  --pretrained_dir="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_32.npz" \
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output



