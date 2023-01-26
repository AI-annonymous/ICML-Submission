# ELens
# CUB
# VIT - Epoch 7
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --train_baseline "y" --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"
# tensor(528) tensor(1048) torch.Size([1183])
# Accuracy g model: 88.58833312988281 %
# Accuracy intervened model: 44.632293701171875 %
# Drop: 49.61832046508789 %
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --seed 1 --test_baseline "y" --g_checkpoint "g_best_model_epoch_16.pth.tar" --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --root-bb "lr_0.03_epochs_95"

# CNN Epoch 30
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --train_baseline "y" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
# Old best val accuracy: 80.89602704987321 (%) || New best val accuracy: 80.98055790363483 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline_PostHoc/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/g_best_model_epoch_83.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --seed 1 --test_baseline "y" --g_checkpoint "g_best_model_epoch_83.pth.tar" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"


#-----------------------------------------------------------------------------------------------------------------------------------
# logistic explainer
# CUB
# VIT - Epoch 7
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --logistic_explainer "y" --train_baseline "y" --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"
# Old best val accuracy: 89.68723584108199 (%) || New best val accuracy: 89.77176669484362 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline_PostHoc/ViT-B_16/Logistic_explainer/g_best_model_epoch_30.pth.tar
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --logistic_explainer "y" --test_baseline "y" --g_checkpoint "g_best_model_epoch_21.pth.tar" --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"


# CNN Epoch 30
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --logistic_explainer "y" --train_baseline "y" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --logistic_explainer "y" --test_baseline "y" --g_checkpoint "g_best_model_epoch_78.pth.tar"  --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --test_baseline "y" --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
# Old best val accuracy: 83.0938292476754 (%) || New best val accuracy: 83.34742180896028 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline_PostHoc/ResNet101/Logistic_explainer/g_best_model_epoch_78.pth.tar

#-----------------------------------------------------------------------------------------------------------------------------------
# ELens
# Awa2
# VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "n" --train_baseline "y" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --seed 1  --test_baseline "y" --g_checkpoint "best_model.pth.tar" --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_700_checkpoint.bin"

# CNN
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "n" --train_baseline "y" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.0001 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --seed 1 --test_baseline "y" --g_checkpoint "best_model.pth.tar" --bs 16  --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.0001 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_47.pth.tar"


#-----------------------------------------------------------------------------------------------------------------------------------
# logistic explainer
# Awa2
# CNN
# VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "y" --train_baseline "y" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"

# CNN
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_awa2.py --logistic_explainer "y" --train_baseline "y" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.0001 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"


#-----------------------------------------------------------------------------------------------------------------------------------
# MIMIC-CXR
# Epoch 2
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--train_baseline "y" \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lambda-lens 0.0001 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


# Epoch 2
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--train_baseline "y" \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lambda-lens 0.0001 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# HAM 10k
# Epoch 3
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --seed 1 --epochs 100  --train_baseline "y" --bs 32 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --arch "Inception_V3"

# SIIM-ISIC
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --seed 1 --dataset "SIIM-ISIC" --epochs 10  --train_baseline "y" --bs 32 --temperature-lens 0.7 --lambda-lens 0.0001 --hidden-nodes 10 --arch "Inception_V3" --lr 0.01 --data-root "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC"


