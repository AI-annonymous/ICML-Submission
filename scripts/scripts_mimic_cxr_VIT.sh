# Pneumonia
# BB
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_mimic_cxr.py \
  --model_type='ViT-B_32_densenet' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumonia" \
  --pretrained="n" \
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"


# BB path
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/mimic_cxr/BB/lr_0.03_epochs_60_loss_CE/ViT-B_32_densenet/pneumonia/n
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/BB/lr_0.03_epochs_60_loss_CE/ViT-B_32_densenet/pneumonia/n
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/mimic_cxr/BB/lr_0.03_epochs_60_loss_CE_ViT-B_32_densenet_pneumonia_n
# Testing
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/test_mimic_cxr.py \
  --model_type='ViT-B_32_densenet' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumonia" \
  --checkpoint-file="VIT_mimic_cxr_7600_checkpoint.bin" \
  --pretrained="n" \
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"

# T
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_7600_checkpoint.bin"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumonia/dataset_g"\
  --network-type "VIT"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_7600_checkpoint.bin"\
  --flattening-type="vit_flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumonia/dataset_g"\
  --network-type "VIT"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_7600_checkpoint.bin"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_12.pth.tar"\
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumonia/dataset_g"\
  --network-type "VIT"





# Pneumothorax
# BB
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_mimic_cxr.py \
  --model_type='ViT-B_32_densenet' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumothorax" \
  --pretrained="n" \
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# BB path
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/mimic_cxr/BB/lr_0.03_epochs_60_loss_CE/ViT-B_32_densenet/pneumothorax/n
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/BB/lr_0.03_epochs_60_loss_CE/ViT-B_32_densenet/pneumothorax/n
# test_mimic_cxr
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/test_mimic_cxr.py \
  --model_type='ViT-B_32_densenet' \
  --learning_rate=0.03 \
  --resize=512 \
  --loss="CE" \
  --selected-obs="pneumothorax" \
  --checkpoint-file="VIT_mimic_cxr_8200_checkpoint.bin" \
  --pretrained="n" \
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# T
# Training scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_8200_checkpoint.bin"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumothorax/dataset_g"\
  --network-type "VIT"



  python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_8200_checkpoint.bin"\
  --flattening-type="vit_flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumothorax/dataset_g"\
  --network-type "VIT"


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='ViT-B_32_densenet' \
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
  --bb-chkpt-folder="lr_0.03_epochs_60_loss_CE"\
  --checkpoint-bb="VIT_mimic_cxr_8200_checkpoint.bin"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"\
  --feature-path "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/pneumothorax/dataset_g"\
  --network-type "VIT"


