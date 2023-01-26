# BB
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="pneumonia" \
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"

 python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="pneumothorax" \
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"

  python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"


  python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
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
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"

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
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"


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
  --checkpoint-t="g_best_model_epoch_16.pth.tar"\
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
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
  --checkpoint-t="g_best_model_epoch_16.pth.tar"\
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"> $slurm_output



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
--dataset "mimic_cxr" \
--cov 0.1 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--dataset "mimic_cxr" \
--cov 0.1 0.1 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--checkpoint-model "model_seq_epoch_237.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"

# train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--dataset "mimic_cxr" \
--cov 0.1 0.1 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 32.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--expert-to-train "residual" \
--checkpoint-model "model_seq_epoch_237.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"


# test residual






#######################################
# Pneumothorax
#######################################
#######################################
# iter1
#######################################
# train g (Epoch 120)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--dataset "mimic_cxr" \
--cov 0.25 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 64.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# test g
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
--iter 1 \
--dataset "mimic_cxr" \
--cov 0.25 0.25 \
--bs 64 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 64.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--expert-to-train "explainer" \
--checkpoint-model "model_seq_epoch_158.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
--iter 1 \
--dataset "mimic_cxr" \
--cov 0.25 0.25 \
--bs 16 \
--dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
--input-size-pi 2048 \
--optim "SGD" \
--lr 0.1 0.1 \
--temperature-lens 7.6 \
--lm 64.0 \
--lambda-lens 0.0001 \
--alpha-KD 0.99 \
--temperature-KD 20 \
--hidden-nodes 20 20 \
--layer "layer4" \
--bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
--checkpoint-bb "g_best_model_epoch_4.pth.tar" \
--expert-to-train "residual" \
--checkpoint-model "model_seq_epoch_158.pth.tar" \
--arch "densenet121" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"
# test residual
