#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/interpretable/%j_bash_run_BB.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/interpretable/run_BB_$CURRENT.out
echo "AWA2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2.py > $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_baseline_explainer "y" --train_baseline_backbone "n" --arch "ResNet50" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/Baseline/lr_0.001_epochs_95/ResNet50" --bs 10 --bb-chkpt-file "g_best_model_epoch_13.pth.tar"  > $slurm_output

