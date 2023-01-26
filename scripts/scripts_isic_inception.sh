# Train test BB
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_ISIC.py --optim "SGD"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 1 --checkpoint-model "model_g_best_model_epoch_20.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# ------------------- Metrics ---------------------
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3/g_best_model_epoch_4.pth.tar
# Epoch: [4/95] Train_loss: 0.1367 Train_Accuracy: 94.65 (%) Val_loss: 0.4164 Val_Accuracy: 85.0 (%) Best_Val_Accuracy: 85.0 (%)  Val_AUROC: 0.875 (%)  Val_AURPC: 0.6244 (%)  Epoch_Duration: 109.1572
# ------------------- Metrics ---------------------

#---------------------------------
# # iter 1 
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 1 --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 1 --checkpoint-model "model_g_best_model_epoch_20.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# ------------------- Metrics ---------------------
# Accuracy of the network: 98.34710743801654 (%)
# Val AUROC of the network: 0.9411764705882353 (0-1)
# ------------------- Metrics ---------------------
# Output sizes: 
# tensor_images size: torch.Size([121, 3, 299, 299])
# tensor_concepts size: torch.Size([121, 8])
# tensor_preds size: torch.Size([121, 2])
# tensor_preds_bb size: torch.Size([121, 2])
# tensor_y size: torch.Size([121])
# tensor_conceptizator_concepts size: torch.Size([1, 121, 8])



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 1 --checkpoint-model "model_g_best_model_epoch_20.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 1 --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# Accuracy of Residual: 88.8888931274414 || Accuracy of BB: 88.3720932006836


#---------------------------------
# # iter 2
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1/" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# Output sizes: 
# tensor_images size: torch.Size([77, 3, 299, 299])
# tensor_concepts size: torch.Size([77, 8])
# tensor_preds size: torch.Size([77, 2])
# tensor_preds_bb size: torch.Size([77, 2])
# tensor_y size: torch.Size([77])
# tensor_conceptizator_concepts size: torch.Size([1, 77, 8])
# Model-specific sizes: 
# tensor_concept_mask size: torch.Size([2, 8])
# tensor_alpha size: torch.Size([2, 8])
# tensor_alpha_norm size: torch.Size([2, 8])
# ------------------- Metrics ---------------------
# Accuracy of the network: 94.8051948051948 (%)
# Val AUROC of the network: 0.7431506849315067 (0-1)
# ------------------- Metrics ---------------------



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# Accuracy of Residual: 83.5016860961914 || Accuracy of BB: 85.52188110351562

---------------------------------
# # iter 3
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Output sizes: 
# tensor_images size: torch.Size([108, 3, 299, 299])
# tensor_concepts size: torch.Size([108, 8])
# tensor_preds size: torch.Size([108, 2])
# tensor_preds_bb size: torch.Size([108, 2])
# tensor_y size: torch.Size([108])
# tensor_conceptizator_concepts size: torch.Size([1, 108, 8])
# Model-specific sizes: 
# tensor_concept_mask size: torch.Size([2, 8])
# tensor_alpha size: torch.Size([2, 8])
# tensor_alpha_norm size: torch.Size([2, 8])
# ------------------- Metrics ---------------------
# Accuracy of the network: 80.55555555555556 (%)
# Val AUROC of the network: 0.8661448140900195 (0-1)
# ------------------- Metrics ---------------------


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# Accuracy of Residual: 84.65909576416016 || Accuracy of BB: 85.2272720336914


#---------------------------------
# # iter 4
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"


# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar"  --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Output sizes: 
# tensor_images size: torch.Size([33, 3, 299, 299])
# tensor_concepts size: torch.Size([33, 8])
# tensor_preds size: torch.Size([33, 2])
# tensor_preds_bb size: torch.Size([33, 2])
# tensor_y size: torch.Size([33])
# tensor_conceptizator_concepts size: torch.Size([1, 33, 8])
# Model-specific sizes: 
# tensor_concept_mask size: torch.Size([2, 8])
# tensor_alpha size: torch.Size([2, 8])
# tensor_alpha_norm size: torch.Size([2, 8])
# ------------------- Metrics ---------------------
# Accuracy of the network: 84.84848484848484 (%)
# Val AUROC of the network: 0.8 (0-1)


# 
# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"
# Accuracy of Residual: 82.55033874511719 || Accuracy of BB: 83.22147369384766

#---------------------------------
# # iter 5
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 5  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"
# Output sizes: 
# tensor_images size: torch.Size([116, 3, 299, 299])
# tensor_concepts size: torch.Size([116, 8])
# tensor_preds size: torch.Size([116, 2])
# tensor_preds_bb size: torch.Size([116, 2])
# tensor_y size: torch.Size([116])
# tensor_conceptizator_concepts size: torch.Size([1, 116, 8])
# Model-specific sizes: 
# tensor_concept_mask size: torch.Size([2, 8])
# tensor_alpha size: torch.Size([2, 8])
# tensor_alpha_norm size: torch.Size([2, 8])
# ------------------- Metrics ---------------------
# Accuracy of the network: 62.93103448275862 (%)
# Val AUROC of the network: 0.65123362777947 (0-1)
# ------------------- Metrics ---------------------


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



#---------------------------------
# # iter 6
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 6  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 6 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" "model_g_best_model_epoch_1.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "explainer" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"
Output sizes: 
tensor_images size: torch.Size([16, 3, 299, 299])
tensor_concepts size: torch.Size([16, 8])
tensor_preds size: torch.Size([16, 2])
tensor_preds_bb size: torch.Size([16, 2])
tensor_y size: torch.Size([16])
tensor_conceptizator_concepts size: torch.Size([1, 16, 8])
Model-specific sizes: 
tensor_concept_mask size: torch.Size([2, 8])
tensor_alpha size: torch.Size([2, 8])
tensor_alpha_norm size: torch.Size([2, 8])
------------------- Metrics ---------------------

Accuracy of the network: 87.5 (%)
Val AUROC of the network: 0.8928571428571428 (0-1)

# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ISIC.py --iter 6 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" "model_g_best_model_epoch_1.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset "SIIM-ISIC" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ISIC.py --iter 6 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_20.pth.tar" "model_g_best_model_epoch_18.pth.tar" "model_g_best_model_epoch_9.pth.tar" "model_g_best_model_epoch_21.pth.tar" "model_g_best_model_epoch_19.pth.tar" "model_g_best_model_epoch_1.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset ""SIIM-ISIC"" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"

# Output sizes: 
# tensor_images size: torch.Size([29, 3, 299, 299])
# tensor_concepts size: torch.Size([29, 8])
# tensor_preds_bb size: torch.Size([29, 2])
# tensor_preds_residual size: torch.Size([29, 2])
# tensor_y size: torch.Size([29])
# Accuracy of the residual: 89.65517241379311 (%)
# Val AUROC of the residual: 0.5512820512820513 (0-1)
# Accuracy of Residual: 88.8888931274414 || Accuracy of BB: 88.8888931274414

