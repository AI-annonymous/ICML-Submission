#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/completeness/cub_resnet_moie_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output1=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/completeness/cub_resnet_moie_$CURRENT.out

echo "Awa2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2.py > $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "MoIE" --epochs 75 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output1
