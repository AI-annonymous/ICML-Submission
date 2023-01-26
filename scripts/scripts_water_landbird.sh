# CNN BB Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --img-size 224
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --img-size 224


# CNN BB test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --checkpoint-file "g_best_model_epoch_26.pth.tar" --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --checkpoint-file "g_best_model_epoch_70.pth.tar" --img-size 448

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet50" --checkpoint-file "g_best_model_epoch_51.pth.tar" --img-size 224
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --spurious-waterbird-landbird "y" --bs 16 --arch "ResNet101" --checkpoint-file "g_best_model_epoch_99.pth.tar" --img-size 224


# T CNN Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_26.pth.tar --bs 16 --layer layer4 --flattening-type adaptive --arch ResNet50 --img-size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_70.pth.tar --bs 16 --layer layer4 --flattening-type adaptive --arch ResNet101 --img-size 448


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_51.pth.tar --bs 16 --layer layer4 --flattening-type adaptive --arch ResNet50 --img-size 224
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --epochs 300 --checkpoint-file g_best_model_epoch_99.pth.tar --bs 16 --layer layer4 --flattening-type adaptive --arch ResNet101 --img-size 224

# T CNN Test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-waterbird-landbird "y" --img-size 224 --checkpoint-file "g_best_model_epoch_51.pth.tar" --checkpoint-file-t "g_best_model_epoch_197.pth.tar" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet50"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-waterbird-landbird "y" --img-size 448 --checkpoint-file "g_best_model_epoch_26.pth.tar" --checkpoint-file-t "g_best_model_epoch_200.pth.tar" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet50"

# G CNN iter 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --img-size 224 --spurious-waterbird-landbird "y" --root-bb "lr_0.001_epochs_300" --checkpoint-bb "g_best_model_epoch_51.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.00 --bs 16 --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --img-size 224 --spurious-waterbird-landbird "y" --checkpoint-model "model_g_best_model_epoch_23.pth.tar" --root-bb "lr_0.001_epochs_300" --checkpoint-bb "g_best_model_epoch_51.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.00 --bs 16 --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "adaptive" --arch "ResNet50"

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --img-size 224 --spurious-waterbird-landbird "y" --root-bb "lr_0.001_epochs_300" --checkpoint-bb "g_best_model_epoch_51.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.0 --bs 32 --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50_projected"


# Projected

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_adjust_phi.py --img-size 224 --finetune "n" --disable-batchnorm "n" --spurious-waterbird-landbird "y" --layer "layer4" --arch "ResNet50" --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --checkpoint-bb "g_best_model_epoch_51.pth.tar" --root-bb "lr_0.001_epochs_300" --root-explainer "_cov_1.0/iter1" --bs 32 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_adjust_phi.py --img-size 224 --finetune "y" --disable-batchnorm "n" --spurious-waterbird-landbird "y" --layer "layer4" --arch "ResNet50" --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --checkpoint-bb "g_best_model_epoch_10.pth.tar" --root-bb "lr_0.001_epochs_300" --root-explainer "_cov_1.0/iter1" --bs 32 

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --projected "y" --spurious-waterbird-landbird "y" --epochs 300 --root-explainer "_cov_1.0/iter1/t_projected/batch_norm_n_finetune_y" --bb-projected "_cov_1.0/iter1/bb_projected/batch_norm_n_finetune_y" --checkpoint-file "g_best_model_epoch_10.pth.tar"  --dataset-folder-concepts "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --bs 32 --layer layer4 --flattening-type "projected" --arch "ResNet50" --img-size 224
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --projected "y" --spurious-waterbird-landbird "y" --epochs 300 --root-explainer "_cov_1.0/iter1/t_projected/batch_norm_n_finetune_y" --bb-projected "_cov_1.0/iter1/bb_projected/batch_norm_n_finetune_y" --checkpoint-file "g_best_model_epoch_10.pth.tar"  --checkpoint-file-t "g_best_model_epoch_198.pth.tar" --dataset-folder-concepts "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --bs 32 --layer layer4 --flattening-type "projected" --arch "ResNet50" --img-size 224

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --img-size 224 --projected "y" --spurious-waterbird-landbird "y" --root-bb "lr_0.001_epochs_300" --bb-projected "_cov_1.0/iter1/bb_projected/batch_norm_n_finetune_y" --checkpoint-bb "g_best_model_epoch_10.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.00 --bs 32 --dataset-folder "explainer/ResNet50/_cov_1.0/iter1/t_projected/batch_norm_n_finetune_y/dataset_g" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --img-size 224 --projected "y" --spurious-waterbird-landbird "y" --root-bb "lr_0.001_epochs_300" --checkpoint-model "model_g_best_model_epoch_488.pth.tar" --bb-projected "_cov_1.0/iter1/bb_projected/batch_norm_n_finetune_y" --checkpoint-bb "g_best_model_epoch_10.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 1.00 --bs 32 --dataset-folder "explainer/ResNet50/_cov_1.0/iter1/t_projected/batch_norm_n_finetune_y/dataset_g" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet50"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_adjust_phi.py --use_original "y" --img-size 224 --finetune "y" --disable-batchnorm "n" --spurious-waterbird-landbird "y" --layer "layer4" --arch "ResNet50" --dataset-folder "lr_0.001_epochs_300_ResNet50_layer4_adaptive_sgd_BCE" --checkpoint-bb "g_best_model_epoch_10.pth.tar" --root-bb "lr_0.001_epochs_300" --root-explainer "_cov_1.0/iter1" --bs 32 











# T VIT Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --img-size 448 --lr 0.03 --checkpoint-file "VIT_CUBS_3000_checkpoint.bin" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --spurious-waterbird-landbird "y" --img-size 224 --lr 0.03 --checkpoint-file "VIT_CUBS_1400_checkpoint.bin" --bs 16 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT

# T VIT Test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --spurious-waterbird-landbird "y" --lr 0.03  --img-size 224 --checkpoint-file "VIT_CUBS_1400_checkpoint.bin" --bs 8 --solver-LR "sgd" --loss-LR "BCE" --arch "ViT-B_16" --flattening-type VIT --checkpoint-file-t "g_best_model_epoch_200.pth.tar"


# VIT BB Train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_spurious_BB_CUB.py --img_size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_spurious_BB_CUB.py --img_size 224

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/test_CUB.py --checkpoint-file "VIT_CUBS_3000_checkpoint.bin" --img_size 448
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/test_CUB.py --checkpoint-file "VIT_CUBS_1400_checkpoint.bin" --img_size 224


