#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline_posthoc/%j_awa2_resnet_logistic_explainer_train.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/awa2/baseline_posthoc/awa2_resnet_logistic_explainer_train_$CURRENT.out
echo "Awa2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "n" --train_baseline "y" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16" > $slurm_output
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "y" --train_baseline "y" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.0001 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50" > $slurm_output
