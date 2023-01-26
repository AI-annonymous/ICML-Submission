#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/t/%j_bash_run_explainer.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/t/run_explainer_$CURRENT.out
echo "MIMIC_CXR"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000


####################################################
# # Flattening type flatten - disease pneumonia
####################################################

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --loss1="BCE"\
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
#   --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumonia"\
#   --labels "0 (No Pneumonia)" "1 (Pneumonia)"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='model_seq_epoch_18.pth.tar' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumonia"> $slurm_output

####################################################
# # Flattening type flatten - disease pneumothorax
####################################################
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

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='model_seq_epoch_18.pth.tar' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumothorax"> $slurm_output

##############################################################
# # Flattening type flatten - disease pneumonia pneumothorax
##############################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock4" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock3" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output


####################################################
# # Flattening type adaptive - disease pneumonia
####################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock4" \
#   --selected-obs="pneumonia"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumonia"> $slurm_output

####################################################
# # Flattening type adaptive - disease pneumothorax
####################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock4" \
#   --selected-obs="pneumothorax"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumothorax"> $slurm_output


##############################################################
# # Flattening type adaptive - disease pneumonia pneumothorax
##############################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock4" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="adaptive" \
#   --layer="features_denseblock3" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output


####################################################
# # Flattening type max_pool - disease pneumonia
####################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock4" \
#   --selected-obs="pneumonia"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumonia"> $slurm_output

####################################################
# # Flattening type max_pool - disease pneumothorax
####################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock4" \
#   --selected-obs="pneumothorax"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_3.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumothorax"> $slurm_output


##############################################################
# # Flattening type max_pool - disease pneumonia pneumothorax
##############################################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock4" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock3" \
#   --selected-obs "pneumonia" "pneumothorax"> $slurm_output


##########################################
### Test
##########################################
# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --checkpoint-t="model_seq_epoch_19.pth.tar"\
#   --flattening-type="flatten" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumothorax"> $slurm_output


# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
#   --arch='densenet121' \
#   --workers=5 \
#   --epochs=60 \
#   --start-epoch=0 \
#   --batch-size=16 \
#   --learning-rate=0.01 \
#   --resize=512 \
#   --resume='' \
#   --gpu=0 \
#   --world-size=1 \
#   --rank=0 \
#   --ngpus-per-node=2 \
#   --checkpoint-bb="model_seq_epoch_14.pth.tar"\
#   --checkpoint-t="model_seq_epoch_20.pth.tar"\
#   --flattening-type="max_pool" \
#   --layer="features_denseblock3" \
#   --selected-obs="pneumothorax"> $slurm_output


