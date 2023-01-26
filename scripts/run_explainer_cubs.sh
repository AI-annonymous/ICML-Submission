#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/%j_bash_run_explainer.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/run_explainer_$CURRENT.out
echo "CUB"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000

# Training scripts

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --cov 0.4 --use-concepts-as-pi-input "y" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --cov 0.4 --use-concepts-as-pi-input "y" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --cov 0.4 --use-concepts-as-pi-input "n" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.5 0.4 --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 0.001 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16" > $slurm_output