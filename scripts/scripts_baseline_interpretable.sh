#-----------------------------------------------------------------------------------------------------------------------------------
# Cubs
# Resnet101
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_baseline_backbone "y" --arch "ResNet101"

# Epoch 199 (64 %)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_g "y" --train_baseline_backbone "n" --arch "ResNet101" --train_cem "n" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ResNet101/Backbone" --bb-chkpt-file "best_model.pth.tar"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --g_checkpoint "g_best_model_epoch_398.pth.tar"  --train_g "n" --train_baseline_backbone "n" --arch "ResNet101" --train_cem "n" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ResNet101/Backbone" --bb-chkpt-file "best_model.pth.tar"


# CUBS
# VIT
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_CUB_baseline.py --arch "ViT-B_16"

# Epoch 6 (87 %)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_g "y" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/Backbone" --lr 0.03 --bs 8 --bb-chkpt-file "VIT_CUBS_9200_checkpoint.bin"
# Test and save
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_g "n" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/Backbone" --lr 0.03 --bs 8 --bb-chkpt-file "VIT_CUBS_8700_checkpoint.bin" --g_checkpoint "g_best_model_epoch_12.pth.tar"
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/explainer
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ViT-B_16/explainer
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/cub/Baseline/lr_0.03_epochs_95_ViT-B_16/explainer
#-----------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------
Joint
# Cubs
# Resnet101
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --joint_cbm "y" --lambda_joint 0.01  --train_baseline_backbone "y" --arch "ResNet101"

# Epoch 199 (64 %)


# CUBS
# VIT
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_CUB_baseline.py --joint_cbm "y" --lambda_joint 0.01 --arch "ViT-B_16"
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_cub_baseline_joint.py --lambda_joint 0.01 

# Epoch 6 (87 %)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_g "y" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/Backbone" --lr 0.03 --bs 8 --bb-chkpt-file "VIT_CUBS_9200_checkpoint.bin"
# Test and save
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_cub.py --train_g "n" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/Backbone" --lr 0.03 --bs 8 --bb-chkpt-file "VIT_CUBS_9200_checkpoint.bin" --g_checkpoint "g_best_model_epoch_12.pth.tar"
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/Baseline/ViT-B_16/explainer
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ViT-B_16/explainer
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/cub/Baseline/lr_0.03_epochs_95_ViT-B_16/explainer
#-----------------------------------------------------------------------------------------------------------------------------------


# AWA2
# Resnet101
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --arch "ResNet50" --train_baseline_backbone "y"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_baseline_explainer "y" --train_baseline_backbone "n" --arch "ResNet50" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/Baseline/lr_0.001_epochs_95/ResNet50" --bs 10 --bb-chkpt-file "g_best_model_epoch_13.pth.tar"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_g "n" --train_baseline_backbone "n" --train_baseline_backbone "n" --arch "ResNet50" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/Baseline/lr_0.001_epochs_95/ResNet50" --bs 10 --bb-chkpt-file "g_best_model_epoch_13.pth.tar" --g_checkpoint "g_best_model_epoch_15.pth.tar"

# VIT
# Train concept bank
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2_baseline.py

# Epoch 6 (87 %)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_baseline_explainer "y" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/Baseline/lr_0.03_epochs_95/ViT-B_16" --lr 0.03 --bs 10 --bb-chkpt-file "VIT_CUBS_9500_checkpoint.bin"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_awa2.py --train_g "n" --train_baseline_backbone "n" --train_baseline_backbone "n" --arch "ViT-B_16" --bb-chkpt "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/Baseline/lr_0.03_epochs_95/ViT-B_16" --lr 0.03 --bs 10 --bb-chkpt-file "VIT_CUBS_9500_checkpoint.bin" --g_checkpoint "g_best_model_epoch_55.pth.tar"

#-----------------------------------------------------------------------------------------------------------------------------------
# HAM10K
# InceptionV3
# Epoch 125
# use Mert's numbers 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_ham10k.py

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_ham10k.py --test "y" --checkpoint-model "model_seq_epoch_125.pth.tar"


# model_seq_epoch_10.pth.tar
# Benign: BWV | ~BWV
# Malignant: BWV & IrregularStreaks & ~AtypicalPigmentNetwork & ~RegressionStructures & ~TypicalPigmentNetwork

# model_seq_epoch_100.pth.tar
# Benign: ~RegularStreaks
# Malignant: BWV

# model_seq_epoch_125.pth.tar
# Benign: BWV | ~BWV
# Malignant: (BWV & IrregularStreaks) | (BWV & ~RegressionStructures)


#-----------------------------------------------------------------------------------------------------------------------------------
# ISIC
# InceptionV3
# 
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_ISIC.py

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_ISIC.py --test "y" --checkpoint-model "model_seq_epoch_34.pth.tar"
# Val_Accuracy: 81.6 (%)  Val_AUROC: 0.8441 


---------------------------------------------------------------------------------------------
# MIMIC-CXR
# Train backbone
# Pneumonia
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"

# test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="pneumonia"\
  --labels "0 (No Pneumonia)" "1 (Pneumonia)"

# Explainer
# Epoch 9
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
--train_baseline_backbone="n" \
--epochs 500 \
--batch-size 64 \
--temperature-lens 7.6 \
--lambda-lens 0.0001 \
--hidden-nodes 20 20 \
--arch "densenet121" \
--dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/pneumonia" \
--selected-obs="pneumonia" \
--labels "0 (No Pneumonia)" "1 (Pneumonia)"




# MIMIC-CXR
# Train backbone
# Pneumothorax
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"

# Test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="pneumothorax"\
  --labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# Train explainer
# Epoch 56
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
--train_baseline_backbone="n" \
--epochs 500 \
--batch-size 64 \
--temperature-lens 10.0 \
--lambda-lens 0.0001 \
--hidden-nodes 20 20 \
--arch "densenet121" \
--dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/pneumothorax" \
--selected-obs="pneumothorax" \
--labels "0 (No Pneumothorax)" "1 (Pneumothorax)"


# Train backbone
# Cardiomegaly
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"


# test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --arch='densenet121' \
  --workers=4 \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --train_baseline_backbone="n" \
  --epochs 500 \
  --batch-size 64 \
  --temperature-lens 7.6 \
  --lambda-lens 0.0001 \
  --hidden-nodes 20 20 \
  --arch "densenet121" \
  --dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/cardiomegaly" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"


# Train backbone
# consolidation
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --arch='densenet121' \
  --workers=4 \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"

# test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --train_baseline_backbone="n" \
  --epochs 500 \
  --batch-size 64 \
  --temperature-lens 7.6 \
  --lambda-lens 0.0001 \
  --hidden-nodes 20 20 \
  --arch "densenet121" \
  --dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/consolidation" \
  --selected-obs="consolidation" \
  --labels "0 (No Consolidation)" "1 (Consolidation)"


# Train backbone
# Edema
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --arch='densenet121' \
  --workers=4 \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"

# test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --arch='densenet121' \
  --workers=4 \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --train_baseline_backbone="n" \
  --epochs 500 \
  --batch-size 64 \
  --temperature-lens 7.6 \
  --lambda-lens 0.0001 \
  --hidden-nodes 20 20 \
  --arch "densenet121" \
  --dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/edema" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)"


# Train backbone
# effusion
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"

# test
# Epoch 1
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --arch='densenet121' \
  --workers=4 \
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
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-bb="g_best_model_epoch_1.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_baseline_mimic_cxr.py \
  --train_baseline_backbone="n" \
  --epochs 500 \
  --batch-size 64 \
  --temperature-lens 7.6 \
  --lambda-lens 0.0001 \
  --hidden-nodes 20 20 \
  --arch "densenet121" \
  --dataset-folder-concepts "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/effusion" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"



