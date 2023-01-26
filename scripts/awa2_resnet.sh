#!/bin/sh
#SBATCH --output=path/awa2_resnet_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=awa2_resnet_bb_train_$CURRENT.out
slurm_output_bb_test=awa2_resnet_bb_test_$CURRENT.out
slurm_output_t_train=awa2_resnet_t_train_$CURRENT.out
slurm_output_t_test=awa2_resnet_t_test_$CURRENT.out
slurm_output_iter1_g_train=awa2_resnet_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=awa2_resnet_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=awa2_resnet_iter1_residual_train_$CURRENT.out
slurm_output_iter2_g_train=awa2_resnet_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=awa2_resnet_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=awa2_resnet_iter2_residual_train_$CURRENT.out
slurm_output_iter3_g_train=awa2_resnet_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=awa2_resnet_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=awa2_resnet_iter3_residual_train_$CURRENT.out
slurm_output_iter4_g_train=awa2_resnet_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=awa2_resnet_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=awa2_resnet_iter4_residual_train_$CURRENT.out
slurm_output_iter5_g_train=awa2_resnet_iter5_g_train_$CURRENT.out
slurm_output_iter5_g_test=awa2_resnet_iter5_g_test_$CURRENT.out
slurm_output_iter5_residual_train=awa2_resnet_iter5_residual_train_$CURRENT.out
slurm_output_iter6_g_train=awa2_resnet_iter6_g_train_$CURRENT.out
slurm_output_iter6_g_test=awa2_resnet_iter6_g_test_$CURRENT.out
slurm_output_iter6_residual_train=awa2_resnet_iter6_residual_train_$CURRENT.out
slurm_output_iter6_residual_test=awa2_resnet_iter6_residual_test_$CURRENT.out

echo "awa2-200 ResNet101"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

######## Instructions ################
# Please update the corresponding "paths" 
# Make sure validate the parameters from the Appendix in the paper
## bb_chkpt.pth.tar - Checkpoint of BB
## t_chkpt.pth.tar - Checkpoint of T
## gi.pth.tar - Checkpoint of i^{th} expert
## ri.pth.tar - Checkpoint of i^{th} residual
## gi_checkpoint_path - File path where checkpoint of i^{th} expert will be


# BB model
# BB Training scripts

python ./codebase/train_BB_awa2.py --bs 16 --arch "ResNet101" > $slurm_output_bb_train


# T model 
# train
python ./codebase/train_t_awa2.py --checkpoint-file "bb_chkpt.pth.tar" --bs 32 --layer "layer4" --flattening-type "adaptive" --arch "ResNet101" > $slurm_output_t_train

# Test
python ./codebase/test_t_awa2.py --checkpoint-file "bb_chkpt.pth.tar" --checkpoint-file-t "t_chkpt.pth.tar" --save-concepts True --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"> $slurm_output_t_test


# MoIE Training scripts

#---------------------------------
# # iter 1 
#---------------------------------

python ./codebase/train_explainer_awa2.py --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter1_g_train


python ./codebase/test_explainer_awa2.py --checkpoint-model "g1.pth.tar" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101">  $slurm_output_iter1_g_test


python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 1 --expert-to-train "residual" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output_iter1_residual_train




#---------------------------------
# # iter 2 
#---------------------------------
# Train explainer
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar" --checkpoint-residual "r1.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 2 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" >  $slurm_output_iter2_g_train


# # Test explainer
python ./codebase/test_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" --checkpoint-residual "r1.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 2 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_g_test


# # # Train residual
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" --checkpoint-residual "r1.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 2 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_residual_train


#---------------------------------
# # iter 3 
#---------------------------------
# Train explainer
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar"  "g2.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 3 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter3_g_train


# Test explainer
python ./codebase/test_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" "g3.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 3 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01  --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_g_test


# # # # Train residual
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" "g3.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar"  --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 3 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_residual_train


#---------------------------------
# # iter 4
#---------------------------------
# Train explainer
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar"  "g2.pth.tar" "g3.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar" "r3.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" "g3_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 4 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter4_g_train


# # # Test explainer
python ./codebase/test_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" "g3.pth.tar" "g4.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar" "r3.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" "g3_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 4 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_g_test


# # # # # Train residual
python ./codebase/train_explainer_awa2.py --checkpoint-model "g1.pth.tar" "g2.pth.tar" "g3.pth.tar" "g4.pth.tar" --checkpoint-residual "r1.pth.tar" "r2.pth.tar" "r3.pth.tar" --prev_explainer_chk_pt_folder "g1_checkpoint_path" "g2_checkpoint_path" "g3_checkpoint_path" --root-bb "[bb_path]" --checkpoint-bb "bb_chkpt.pth.tar" --iter 4 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder "[concept_dataset_path]" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_residual_train

