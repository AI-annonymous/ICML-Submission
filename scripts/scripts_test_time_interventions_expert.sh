# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 90 108 
#     concepts      acc_g   acc_g_in  acc_gain
# 0          3  91.150442  93.185841  2.233010
# 1          5  91.150442  94.424779  3.592233
# 2         10  91.150442  95.221239  4.466019
# 3         15  91.150442  96.017699  5.339806
# 4         20  91.150442  96.725664  6.116505
# 5         25  91.150442  96.725664  6.116505
# 6         30  91.150442  97.079646  6.504854
# 7         50  91.150442  97.345133  6.796117
# 8         75  91.150442  97.345133  6.796117



# -----------------------------------------------------
# CUB_ResNet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 90 108 
# concepts      acc_g   acc_g_in  acc_gain
# 0          3  86.538462  87.307692  0.888889
# 1          5  86.538462  87.692308  1.333333
# 2         10  86.538462  88.365385  2.111111
# 3         15  86.538462  89.230769  3.111111
# 4         20  86.538462  89.615385  3.555556
# 5         25  86.538462  90.480769  4.555556
# 6         30  86.538462  90.576923  4.666667
# 7         50  86.538462  91.346154  5.555556
# 8         75  86.538462  91.538462  5.777778


# -----------------------------------------------------
# Awa2_vit
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in  acc_gain
# 0         3  97.572679  97.572679  0.000000
# 1         5  97.572679  97.572679  0.000000
# 2        10  97.572679  97.600903  0.028927
# 3        15  97.572679  97.685577  0.115707
# 4        20  97.572679  97.713802  0.144634
# 5        25  97.572679  97.713802  0.144634
# 6        30  97.572679  97.713802  0.144634
# 7        50  97.572679  97.756139  0.188024
# 8        75  97.572679  97.784364  0.216951

# -----------------------------------------------------
# Awa2_Resnet
# -----------------------------------------------------

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 3 4 --model "MoIE" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 75 85
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in  acc_gain
# 0         3  87.393428  87.952638  1.006178
# 1         5  85.393428  88.719928  1.553398
# 2        10  85.393428  88.946036  1.818182
# 3        15  85.393428  89.398251  2.347749
# 4        20  85.393428  89.518842  2.488967
# 5        25  85.393428  89.835393  2.859665
# 6        30  85.393428  89.850467  2.877317
# 7        50  85.393428  90.167018  3.248014
# 8        75  85.393428  92.498643  3.636364

