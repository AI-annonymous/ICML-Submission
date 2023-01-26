#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/%j_bash_run_BB.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/run_BB_$CURRENT.out
echo "AWA2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2.py > $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_68.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "awa2" --cov 0.4 --bs 30 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output

