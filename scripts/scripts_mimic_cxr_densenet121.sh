# BB
# Training scripts
 python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=18 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=8 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=2 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output



# Testing scripts

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE"\
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --selected-obs="pneumonia" \
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE"\
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --selected-obs="pneumothorax" \
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# T
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=18 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=8 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=2 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output

# Testing scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_16.pth.tar"\
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_16.pth.tar"\
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_10.pth.tar"\
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_7.pth.tar"\
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_11.pth.tar"\
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=1 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_8.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_10.pth.tar"\
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output



# Explainer
#######################################
# Pneumonia
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 193)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


# train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


# test residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

#######################################
# iter2
#######################################
# train g (Epoch)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.001 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--dataset "mimic_cxr" \
--cov 0.3 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.001 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"



# test g
# current folder for ipython: densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4
# current folder for ipython: densenet121_lr_0.001_SGD_temperature-lens_7.6_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" "model_g_best_model_epoch_480.pth.tar" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4
# model_g_best_model_epoch_480.pth.tar
# densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4

# train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "residual" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" "model_g_best_model_epoch_480.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# test residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "residual" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" "model_g_best_model_epoch_480.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


# train g (Epoch)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 3 \
--dataset "mimic_cxr" \
--cov 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" "model_g_best_model_epoch_480.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# Test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 3 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_256.pth.tar" "model_g_best_model_epoch_480.pth.tar" "model_g_best_model_epoch_18.pth.tar" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

#######################################
# Pneumothorax
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 120)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.4 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.4 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.4 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# test residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.4 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"



#######################################
# iter2
#######################################
# train g 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_10.0_cov_0.4_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# test g
# ipython folder: densenet121_lr_0.01_SGD_temperature-lens_10.0_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" "model_g_best_model_epoch_395.pth.tar" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_10.0_cov_0.4_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# train residual
# densenet121_lr_0.01_SGD_temperature-lens_10.0_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4/iter2/g
# model_g_best_model_epoch_395.pth.tar
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 10.0 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_g_best_model_epoch_130.pth.tar" "model_g_best_model_epoch_395.pth.tar" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_10.0_cov_0.4_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"



# Explainer
#######################################
# Cardiomegaly
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 193)
# # iter 1 
#---------------------------------
# Train explainer

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 72.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output

# GT g BB
# (tensor(3190.), tensor(828), tensor(948))
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 72.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--arch "densenet121" \
--checkpoint-model "model_seq_epoch_137.pth.tar" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"


# train_residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 72.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_seq_epoch_137.pth.tar" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 72.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_seq_epoch_137.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


#######################################
# iter2
#######################################
# train g 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_137.pth.tar" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output

# test_g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_137.pth.tar" "model_seq_epoch_9.pth.tar" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_137.pth.tar" "model_seq_epoch_9.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--arch "densenet121" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.15 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_seq_epoch_137.pth.tar" "model_seq_epoch_9.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


# for iter 3
# --prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_256.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \




# #######################################
# # iter3
# #######################################
# # train g 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 3 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.05 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_256.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_137.pth.tar" "model_seq_epoch_9.pth.tar" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 3 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.05 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 1024.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_72.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.15_alpha_0.5_selection-threshold_0.5_lm_256.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_137.pth.tar" "model_seq_epoch_9.pth.tar" "model_seq_epoch_27.pth.tar" \
--arch "densenet121" \
--selected-obs="cardiomegaly" \
--labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"



# Explainer
#######################################
# Consolidation
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 193)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--checkpoint-model "model_seq_epoch_122.pth.tar" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"

# train_residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--checkpoint-model "model_seq_epoch_122.pth.tar" \
--arch "densenet121" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"

# test residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--checkpoint-model "model_seq_epoch_122.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"


#######################################
# iter2
#######################################
# train g 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_122.pth.tar" \
--arch "densenet121" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_122.pth.tar" "model_seq_epoch_16.pth.tar" \
--arch "densenet121" \
--selected-obs="consolidation" \
--labels "0 (No Consolidation)" "1 (Consolidation)"


# Explainer
#######################################
# Edema
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 193)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--checkpoint-model "model_seq_epoch_206.pth.tar" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"


# train_residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_seq_epoch_206.pth.tar" \
--arch "densenet121" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"

# test residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.50 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--checkpoint-model "model_seq_epoch_206.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"


#######################################
# iter2
#######################################
# train g 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_206.pth.tar" \
--arch "densenet121" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_206.pth.tar" "model_seq_epoch_5.pth.tar" \
--arch "densenet121" \
--selected-obs="edema" \
--labels "0 (No Edema)" "1 (Edema)"

# Explainer
#######################################
# Effusion
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 193)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 96.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--checkpoint-model "model_seq_epoch_104.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.5 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 96.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--checkpoint-model "model_seq_epoch_104.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output


iter 2
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.2 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_104.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.2\
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_104.pth.tar" "model_seq_epoch_5.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.2 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_104.pth.tar" "model_seq_epoch_5.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 2 \
--expert-to-train "residual" \
--dataset "mimic_cxr" \
--cov 0.2 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 128.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--arch "densenet121" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_8.pth.tar" \
--checkpoint-model "model_seq_epoch_104.pth.tar" "model_seq_epoch_5.pth.tar" \
--checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output


# iter 3
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 3 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20.0 \
--hidden-nodes 20 20 \
--layer "layer4" \
--metric "auroc" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_104.pth.tar" "model_seq_epoch_5.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"> $slurm_output


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 3 \
--expert-to-train "explainer" \
--dataset "mimic_cxr" \
--cov 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.01 \
--temperature-lens 7.6 \
--lm 256.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--arch "densenet121" \
--prev_chk_pt_explainer_folder "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
--checkpoint-model "model_seq_epoch_104.pth.tar" "model_seq_epoch_5.pth.tar" "model_seq_epoch_2.pth.tar" \
--arch "densenet121" \
--selected-obs="effusion" \
--labels "0 (No Effusion)" "1 (Effusion)"



