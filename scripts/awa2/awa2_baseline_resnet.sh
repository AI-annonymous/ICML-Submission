#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_resnet_backbone_train_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output_backbone=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_resnet_backbone_train_$CURRENT.out
slurm_output_explainer=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_resnet_g_train_$CURRENT.out
slurm_output_results=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_resnet_results_train_$CURRENT.out
echo "awa2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --seed 5 --train_baseline_backbone "y" --epochs 95 --arch "ResNet50" --train_baseline_backbone "y" > $slurm_output_backbone
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_g "y" --train_baseline_backbone "n" --arch "ResNet50"  --bb-chkpt-file "best_model.pth.tar" > $slurm_output