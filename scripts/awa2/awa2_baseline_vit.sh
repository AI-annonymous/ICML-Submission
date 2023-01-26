#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_vit_backbone_train_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output_backbone=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_vit_backbone_train_$CURRENT.out
slurm_output_explainer=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_vit_g_train_$CURRENT.out
slurm_output_results=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline/awa2_vit_results_train_$CURRENT.out
echo "awa2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2_baseline.py --seed 5 --num_steps 10000 > $slurm_output_backbone
