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

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_mimic_cxr.py \
  --model_type='ViT-B_16' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumonia" \
  --pretrained_dir="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_16.npz"
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --loss="CE"\
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --selected-obs="pneumothorax" \
#   --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output




