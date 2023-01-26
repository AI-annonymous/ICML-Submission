# BB CNN 

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-specific-classes "y" --bs 16 --arch "ResNet101" 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --img-size 224
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --img-size 224


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-specific-classes "y" --bs 16 --arch "ResNet101" --checkpoint-file "g_best_model_epoch_39.pth.tar"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --checkpoint-file "g_best_model_epoch_26.pth.tar" --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --checkpoint-file "g_best_model_epoch_70.pth.tar" --img-size 448

# BB VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_spurious_BB_CUB.py --spurious-specific-classes "y" --seed 42 --bs 16 --name "VIT_CUBS" --learning_rate 0.03 --arch "ViT-B_16" --eval_every 10 --lr 0.03 --weight-decay 0 --num_steps 10000 --decay_type "cosine" --warmup_steps 500 --max_grad_norm 1.0 --seed 42 --gradient_accumulation_steps 1 --split non-overlap
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --arch "ViT-B_16_projected"   --spurious-specific-classes "y" --checkpoint-file "VIT_CUBS_7000_checkpoint.bin" --seed 42 --bs 8 --learning-rate 0.03 

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_spurious_BB_CUB.py --spurious-specific-classes "y" --seed 42 --bs 8 --name "VIT_CUBS" --learning_rate 0.03 --arch "ViT-B_16" --eval_every 10 --lr 0.03 --weight-decay 0 --num_steps 10000 --decay_type "cosine" --warmup_steps 500 --max_grad_norm 1.0 --seed 42 --gradient_accumulation_steps 1 --split non-overlap


# T VIT Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-specific-classes "y" --lr 0.03 --checkpoint-file "VIT_CUBS_7000_checkpoint.bin" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --img-size 224 --lr 0.03 --checkpoint-file "VIT_CUBS_200_checkpoint.bin" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-specific-classes "y" --lr 0.03 --checkpoint-file "VIT_CUBS_7000_checkpoint.bin" --bs 8 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16_projected" --flattening-type VIT 

# T CNN Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-specific-classes "y" --checkpoint-file g_best_model_epoch_39.pth.tar --bs 32 --layer layer4 --flattening-type adaptive --arch ResNet101

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_26.pth.tar --bs 32 --layer layer4 --flattening-type adaptive --arch ResNet50 --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_70.pth.tar --bs 32 --layer layer4 --flattening-type adaptive --arch ResNet101 --img-size 448






# T VIT Test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-specific-classes "y" --lr 0.03 --checkpoint-file "VIT_CUBS_7000_checkpoint.bin" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT --checkpoint-file-t "g_best_model_epoch_200.pth.tar"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-specific-classes "y" --lr 0.03 --checkpoint-file "VIT_CUBS_7000_checkpoint.bin" --bs 8 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16_projected" --flattening-type VIT --checkpoint-file-t "g_best_model_epoch_200.pth.tar"

# T CNN Test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-specific-classes "y" --checkpoint-file "g_best_model_epoch_39.pth.tar" --checkpoint-file-t "g_best_model_epoch_200.pth.tar" --bs 32 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --projected "y" --bs 16 --spurious-specific-classes "y" --checkpoint-file "g_best_model_epoch_39.pth.tar" --checkpoint-file-t "g_best_model_epoch_200.pth.tar" --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-waterbird-landbird "y" --checkpoint-file "g_best_model_epoch_6.pth.tar" --checkpoint-file-t "g_best_model_epoch_200.pth.tar" --bs 32 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet50"


# CNN
# iter 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious-specific-classes "y" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_39.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.95 --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious-specific-classes "y" --checkpoint-model "model_g_best_model_epoch_46.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_39.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.95 --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_adjust_phi.py --spurious-specific-classes "y" --layer "layer4" --arch "ResNet101" --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --checkpoint-bb "g_best_model_epoch_39.pth.tar" --root-bb "lr_0.001_epochs_95" --bs 16

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious-specific-classes "y" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_39.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious-specific-classes "y" --checkpoint-model "model_g_best_model_epoch_28.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_39.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 16 --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"


# VIT
# iter 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious-specific-classes "y" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_7000_checkpoint.bin" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious-specific-classes "y" --seed 42 --checkpoint-model "model_g_best_model_epoch_18.pth.tar" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_7000_checkpoint.bin" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0  --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_adjust_phi.py --spurious-specific-classes "y" --layer "VIT" --arch "ViT-B_16" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --checkpoint-bb "VIT_CUBS_7000_checkpoint.bin" --root-bb "lr_0.03_epochs_95"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious-specific-classes "y" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_7000_checkpoint.bin" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 8 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16_projected"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious-specific-classes "y" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_7000_checkpoint.bin" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 8 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16_projected"


# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious "y" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8600_checkpoint.bin" --iter 1 --expert-to-train "residual" --dataset "cub" --cov 0.2 --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"


# WaterBird-LandBird
# CNN
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious-waterbird-landbird "y" --root-bb "lr_0.001_epochs_300" --checkpoint-bb "g_best_model_epoch_6.pth.tar" --checkpoint-t "g_best_model_epoch_200.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.99 --bs 16 --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious-waterbird-landbird "y" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" --root-bb "lr_0.001_epochs_300" --checkpoint-bb "g_best_model_epoch_6.pth.tar" --checkpoint-t "g_best_model_epoch_200.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.99 --bs 16 --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"



#---------------------------------
# # iter 2 
#---------------------------------
# lr 0.001
# cov 0.4
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --spurious "y" --checkpoint-model "model_g_best_model_epoch_18.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8600_checkpoint.bin" --iter 2 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.95 --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16" 


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --spurious "y" --checkpoint-model "model_g_best_model_epoch_18.pth.tar" --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8600_checkpoint.bin" --iter 2 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.95 --bs 16 --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 6.0 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "ViT-B_16"






















































