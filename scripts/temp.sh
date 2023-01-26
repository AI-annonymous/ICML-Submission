#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/%j_bash_run_BB.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/run_BB_$CURRENT.out
echo "CIFAR10"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# train BB
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/temp.py > $slurm_output
