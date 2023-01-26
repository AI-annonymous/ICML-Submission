#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/out_bb_mimic_cxr/%j_bash_run_explainer.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T_pneumonia"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/out_bb_mimic_cxr/run_explainer_$CURRENT.out
echo "MIMIC_CXR"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7

# conda activate python_3_7_rtx_6000


# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --selected-obs 'pneumonia' 'pneumothorax' > $slurm_output 

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE"\
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --selected-obs="pneumonia" \
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

  # python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
  # --arch='densenet121' \
  # --workers=5 \
  # --epochs=60 \
  # --start-epoch=0 \
  # --batch-size=16 \
  # --learning-rate=0.01 \
  # --resize=512 \
  # --resume='' \
  # --loss="CE"\
  # --gpu=0 \
  # --world-size=1 \
  # --rank=0 \
  # --ngpus-per-node=2 \
  # --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  # --selected-obs="pneumothorax" \
  # --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --selected-obs="pneumothorax" > $slurm_output 


# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --selected-obs="pneumonia" > $slurm_output 




