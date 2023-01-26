# All
# Cardiomegaly
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_mimic_cxr.py --disease_folder "cardiomegaly" --g_lr 0.01 --hidden-nodes 1000 --dataset "mimic_cxr" --arch "densenet121" --epochs 100 --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" --checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/cardiomegaly" --flattening-type "flatten" --root-bb "lr_0.01_epochs_60_loss_CE" --checkpoint-file-t "g_best_model_epoch_10.pth.tar" --checkpoint-bb "g_best_model_epoch_4.pth.tar" --layer "features_denseblock4"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_mimic_cxr.py --disease_folder "effusion" --g_lr 0.01 --hidden-nodes 1000 --dataset "mimic_cxr" --arch "densenet121" --epochs 100 --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" --checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/effusion" --flattening-type "flatten" --root-bb "lr_0.01_epochs_60_loss_CE" --checkpoint-file-t "g_best_model_epoch_10.pth.tar" --checkpoint-bb "g_best_model_epoch_8.pth.tar" --layer "features_denseblock4"


# HAM 10k
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_ham10k.py --g_lr 0.01 --hidden-nodes 500 --dataset "HAM10k" --arch "Inception_V3" --epochs 100
# Old best val accuracy: 90.01497753369945 (%) || New best val accuracy: 90.26460309535696 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/completeness/g_all/g_best_model_epoch_49.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_ham10k.py --g_checkpoint "g_best_model_epoch_49.pth.tar" --g_lr 0.01 --hidden-nodes 500 --dataset "HAM10k" --arch "Inception_V3" --epochs 100
# Accuracy of the bb: 89.46580129805291 (%)
# Accuracy using the completeness: 90.26460309535696 (%)
# Completeness_score based on accuracy: 1.0202403542061989
# Auroc of the bb: 0.9601720864383156
# Auroc using the completeness: 0.9173654019923102
# Completeness_score based on auroc: 0.906976790406118


# ISIC
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_ham10k.py --g_lr 0.01 --hidden-nodes 500 --dataset "SIIM-ISIC" --arch "Inception_V3" --data-root "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC" --bb-dir "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3" --model-name "g_best_model_epoch_4" --epochs 100
# Old best val accuracy: 84.39999999999999 (%) || New best val accuracy: 84.8 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/completeness/g_all/g_best_model_epoch_12.pth.tar


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_ham10k.py --g_checkpoint "g_best_model_epoch_12.pth.tar" --g_lr 0.01 --hidden-nodes 500 --dataset "SIIM-ISIC" --arch "Inception_V3" --data-root "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC" --bb-dir "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3" --model-name "g_best_model_epoch_4" --epochs 100
# Accuracy of the bb: 85.0 (%)
# Accuracy using the completeness: 84.8 (%)
# Completeness_score based on accuracy: 0.9942857142857143
# Auroc of the bb: 0.874975
# Auroc using the completeness: 0.8446374999999999
# Completeness_score based on auroc: 0.9190946063070871
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/SIIM-ISIC/completeness/Inception_V3/g_all/out_put_predict_bb.pt


# Awa2 Resnet50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_awa2.py --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "g_best_model_epoch_199.pth.tar" --checkpoint-bb "g_best_model_epoch_47.pth.tar" --dataset "awa2" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet50"
# Old best val accuracy: 84.96985934360349 (%) || New best val accuracy: 85.10381781647689 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/completeness/g_all/g_best_model_epoch_57.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_awa2.py --g_checkpoint "g_best_model_epoch_57.pth.tar" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "g_best_model_epoch_199.pth.tar" --checkpoint-bb "g_best_model_epoch_47.pth.tar" --dataset "awa2" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet50"
# Accuracy of the bb: 91.66778298727395 (%)
# Accuracy using the completeness: 85.10381781647689 (%)
# Completeness_score: 0.842469056421797



# Awa2 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_awa2.py --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_39.pth.tar" --checkpoint-bb "VIT_CUBS_700_checkpoint.bin" --dataset "awa2"  --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16"
# Old best val accuracy: 97.30743469524448 (%) || New best val accuracy: 97.3476222371065 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/completeness/ViT-B_16/g_all/g_best_model_epoch_10.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_awa2.py --g_checkpoint "g_best_model_epoch_10.pth.tar" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_39.pth.tar" --checkpoint-bb "VIT_CUBS_700_checkpoint.bin" --dataset "awa2"  --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16"
# Accuracy of the bb: 98.24514400535834 (%)
# Accuracy using the completeness: 97.3476222371065 (%)
# Completeness_score: 0.9813966402887686

# CUB200 Resnet 101
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_cub.py --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "best_model_epoch_62.pth.tar" --checkpoint-bb "best_model_epoch_63.pth.tar" --dataset "cub" --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet101"
# Old best val accuracy: 78.61369399830939 (%) || New best val accuracy: 79.374471682164 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/completeness/g_all/g_best_model_epoch_73.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_cub.py --g_checkpoint "g_best_model_epoch_73.pth.tar" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "best_model_epoch_62.pth.tar" --checkpoint-bb "best_model_epoch_63.pth.tar" --dataset "cub" --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet101"
# Accuracy of the bb: 88.67286559594251 (%)
# Accuracy using the completeness: 79.374471682164 (%)
# Completeness_score: 0.7595628415300547


# CUB200 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_cub.py --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_54.pth.tar" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --dataset "cub" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16"
# Old best val accuracy: 85.96787827557058 (%) || New best val accuracy: 87.06677937447168 (%) , and new model saved..
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/completeness/ViT-B_16/g_all/g_best_model_epoch_35.pth.tar

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_cub.py --g_checkpoint "g_best_model_epoch_35.pth.tar" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_54.pth.tar" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --dataset "cub" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16"
# Accuracy of the bb: 91.71597633136095 (%)
# Accuracy using the completeness: 86.98224852071006 (%)
# Completeness_score: 0.8865248226950354
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ViT-B_16/g_all/out_put_predict_bb.pt

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<========================================================================================================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Get important concepts
# CUB200 Renset101
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "cub" --arch "ResNet101" --iter 6

# CUB200 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "cub" --arch "ViT-B_16" --iter 6


# Awa2 Renset50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "awa2" --arch "ResNet50" --iter 4
Total_size: 7465

# Awa2 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "awa2" --arch "ViT-B_16" --iter 5
Total_size: 7465


# HAM 10k
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "HAM10k" --arch "Inception_V3" --iter 6
Total_size: 2003

# SIIM-ISIC
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "SIIM-ISIC" --arch "Inception_V3" --iter 6
Total_size: 500
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<========================================================================================================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Per Iter Completeness
# CUB200 Renset101
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_cub.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "best_model_epoch_62.pth.tar" --checkpoint-bb "best_model_epoch_63.pth.tar" --dataset "cub" --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet101" --temperature-lens 0.7 --alpha-KD 0.9 --epochs 50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_cub.py --g_checkpoint "g_best_model_epoch_49.pth.tar" --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "best_model_epoch_62.pth.tar" --checkpoint-bb "best_model_epoch_63.pth.tar" --dataset "cub" --dataset-folder "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet101"
# Accuracy of the bb: 88.75 (%)
# Accuracy using the completeness: 78.26923076923077 (%)
# Completeness_score: 0.74728535980149
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ResNet101/g_per_iter/out_put_predict_bb.pt


# CUB200 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_cub.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_54.pth.tar" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --dataset "cub" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16" --temperature-lens 6.0 --alpha-KD 0.99 --epochs 50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_cub.py --g_checkpoint "g_best_model_epoch_36.pth.tar" --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_54.pth.tar" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --dataset "cub" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16" --temperature-lens 6.0 --alpha-KD 0.99 --epochs 50
# Accuracy of the bb: 91.41592920353983 (%)
# Accuracy using the completeness: 86.60176991150442 (%)
# Completeness_score: 0.8754700854700853
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ViT-B_16/g_per_iter/out_put_predict_bb.pt

# Awa2 Renset50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_awa2.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "g_best_model_epoch_199.pth.tar" --checkpoint-bb "g_best_model_epoch_47.pth.tar" --dataset "awa2" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet50" --temperature-lens 0.7 --alpha-KD 0.9 --epochs 100
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_awa2.py --g_checkpoint "g_best_model_epoch_78.pth.tar" --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --checkpoint-t-path "lr_0.001_epochs_95/lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --flattening-type "adaptive" --root-bb "lr_0.001_epochs_95" --checkpoint-file-t "g_best_model_epoch_199.pth.tar" --checkpoint-bb "g_best_model_epoch_47.pth.tar" --dataset "awa2" --dataset-folder "lr_0.001_epochs_95_ResNet50_layer4_adaptive_sgd_BCE" --layer "layer4" --arch "ResNet50" --temperature-lens 0.7 --alpha-KD 0.9 --epochs 100
# Accuracy of the bb: 91.82996683750378 (%)
# Accuracy using the completeness: 82.10732589689479 (%)
# Completeness_score: 0.8275675675675676
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/completeness/ResNet50/g_per_iter/out_put_predict_bb.pt




# Awa2 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_awa2.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_39.pth.tar" --checkpoint-bb "VIT_CUBS_700_checkpoint.bin" --dataset "awa2"  --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16" --temperature-lens 6.0 --alpha-KD 0.99 --epochs 100
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_awa2.py --g_checkpoint "g_best_model_epoch_34.pth.tar"  --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_39.pth.tar" --checkpoint-bb "VIT_CUBS_700_checkpoint.bin" --dataset "awa2"  --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16" --temperature-lens 6.0 --alpha-KD 0.99 --epochs 100
# Accuracy using the completeness: 85.44549836728346 (%)
# Completeness_score: 0.834730056406124
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/completeness/ViT-B_16/g_per_iter/out_put_predict_bb.pt


# HAM10k
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_ham10k.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 500 --dataset "HAM10k" --arch "Inception_V3" --epochs 50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_ham10k.py --g_checkpoint "g_best_model_epoch_17.pth.tar" --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 500 --dataset "HAM10k" --arch "Inception_V3" --epochs 50
# Accuracy of the bb: 92.01541850220264 (%)
# Accuracy using the completeness: 88.98678414096916 (%)
# Completeness_score based on accuracy: 0.9279161205766712
# Auroc of the bb: 0.968354951667611
# Auroc using the completeness: 0.7997319443023189
# Completeness_score based on auroc: 0.8658665305785805
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/completeness/Inception_V3/g_per_iter/out_put_predict_bb.pt


# SIIM-ISIC
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_ham10k.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 500 --dataset "SIIM-ISIC" --arch "Inception_V3" --data-root "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC" --bb-dir "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3" --model-name "g_best_model_epoch_4" --epochs 50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_completeness_ham10k.py --per_iter_completeness "y" --g_lr 0.01 --hidden-nodes 500 --dataset "SIIM-ISIC" --arch "Inception_V3" --data-root "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC" --bb-dir "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3" --model-name "g_best_model_epoch_4" --epochs 50
# Completeness_score based on auroc: 0.8458665305785805














<<<<<<<<<<<<<<<<<<<<<<<<<<<<<========================================================================================================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
################################################### Baseline ###################################################
# Get important concepts
# CUB200 Renset101

# CUB200 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "cub" --arch "ViT-B_16" --baseline "y"


# Awa2 Renset50
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "awa2" --arch "ResNet50" --baseline "y"
Total_size: 7465

# Awa2 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "awa2" --arch "ViT-B_16" --baseline "y"
Total_size: 7465


# HAM 10k
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "HAM10k" --arch "Inception_V3" --baseline "y"
Total_size: 2003

# SIIM-ISIC
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/get_important_concept_masks.py --dataset "SIIM-ISIC" --arch "Inception_V3" --baseline "y"
Total_size: 500
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<========================================================================================================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# CUB200 VIT
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_completeness_cub.py --baseline "y" --g_lr 0.01 --hidden-nodes 1000 --root-bb "lr_0.03_epochs_95" --checkpoint-t-path "lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --flattening-type "VIT" --checkpoint-file-t "g_best_model_epoch_54.pth.tar" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --dataset "cub" --dataset-folder "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --layer "VIT" --arch "ViT-B_16" --temperature-lens 6.0 --alpha-KD 0.99 --epochs 50 --dataset_path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline_PostHoc/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none"
# 0.6
