#-----------------------------------------------------------------------------------------------------------------------------------
# Cubs
# Resnet101
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --arch "ResNet101"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_baseline_backbone "n" --arch "ResNet101"


# VIT
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_CUB_baseline.py

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/lr_0.03_epochs_95/ViT-B_16" --lr 0.03 --bs 8 --bb-chkpt-file "VIT_CUBS_8800_checkpoint.bin"

#-----------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------
# HAM10K
# InceptionV3
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_ham10k.py