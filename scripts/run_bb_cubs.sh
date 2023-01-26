#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/VIT/%j_bash_run_BB.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/VIT/run_BB_$CURRENT.out
echo "CUB"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# Training scripts
#python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --bs 32 --arch "ResNet50"> $slurm_output

#python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --bs 16 --arch "ResNet101"> $slurm_output

# VIT
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --bs 16 --arch "ViT-B_16" --eval_every 100 --lr 0.03 --weight-decay 0 --num_steps 10000 --decay_type "cosine" --warmup_steps 500 --max_grad_norm 1.0 --seed 42 --gradient_accumulation_steps 1 --split non-overlap> $slurm_output


# Testing scripts
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --checkpoint-file "best_model_epoch_63.pth.tar" --save-activations True --layer "layer4" --bs 16 --arch "ResNet101"> $slurm_output

#python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --checkpoint-file "best_model_epoch_63.pth.tar" --save-activations True --layer "layer3" --bs 16 --arch "ResNet101"> $slurm_output
#
#python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --checkpoint-file "best_model_epoch_77.pth.tar" --save-activations True --layer "layer3" --bs 32 --arch "ResNet50"> $slurm_output
#
#python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --checkpoint-file "best_model_epoch_77.pth.tar" --save-activations True --layer "layer4" --bs 32 --arch "ResNet50"> $slurm_output

# train t model
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t.py --checkpoint-file "best_model_epoch_77.pth.tar" --bs 32 --solver-LR "sgd" --loss-LR "BCE" --layer "layer3" --flattening-type "adaptive" --arch "ResNet50"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t.py --checkpoint-file "best_model_epoch_63.pth.tar" --bs 32 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"> $slurm_output


# test t model

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t.py --checkpoint-file "best_model_epoch_63.pth.tar" --checkpoint-file-t "best_model_epoch_50.pth.tar" --save-concepts True --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer3" --flattening-type "flatten" --arch "ResNet101"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t.py --checkpoint-file "best_model_epoch_77.pth.tar" --checkpoint-file-t "best_model_epoch_50.pth.tar" --save-concepts True --bs 32 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "flatten" --arch "ResNet50"> $slurm_output



CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset cub --num_steps 10000 --name VIT_CUB