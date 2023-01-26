# train BB
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CIFAR_10.py 

#---------------------------------
# # iter 1 
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --iter 1 --expert-to-train "explainer" --cov 0.35 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CIFAR10.py --iter 1 --checkpoint-model "model_g_best_model_epoch_376.pth.tar" --expert-to-train "explainer" --cov 0.35 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --iter 1 --checkpoint-model "model_g_best_model_epoch_376.pth.tar" --expert-to-train "residual" --cov 0.35 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64



#---------------------------------
# # iter 2
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" --iter 2 --expert-to-train "explainer" --cov 0.35 0.30 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" --iter 2 --expert-to-train "explainer" --cov 0.35 0.3 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" --iter 2 --expert-to-train "residual" --cov 0.35 0.3 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64



#---------------------------------
# # iter 3
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" --iter 3 --expert-to-train "explainer" --cov 0.35 0.30 0.30 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" "model_g_best_model_epoch_353.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" --iter 3 --expert-to-train "explainer" --cov 0.35 0.30 0.30 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" "model_g_best_model_epoch_353.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" --iter 3 --expert-to-train "residual" --cov 0.35 0.30 0.30 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64



#---------------------------------
# # iter 4
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" "model_g_best_model_epoch_353.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter3" --iter 4 --expert-to-train "explainer" --cov 0.35 0.30 0.30 0.30 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" "model_g_best_model_epoch_353.pth.tar" "model_g_best_model_epoch_84.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter3" --iter 4 --expert-to-train "explainer" --cov 0.35 0.30 0.30 0.30 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CIFAR10.py --checkpoint-model "model_g_best_model_epoch_376.pth.tar" "model_g_best_model_epoch_139.pth.tar" "model_g_best_model_epoch_353.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/CIFAR10/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_1024_cov_0.35_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1/cov_0.3/iter2" --iter 4 --expert-to-train "residual" --cov 0.35 0.30 0.30 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.99 --temperature-KD 10 --hidden-nodes 10 --lm 64
