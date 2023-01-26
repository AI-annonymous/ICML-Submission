# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 
# concepts      acc_g   acc_g_in   acc_drop
# 0         3  91.150442  76.991150  15.533981
# 1         5  91.150442  66.106195  27.475728
# 2        10  91.150442  42.654867  53.203883
# 3        15  91.150442  19.469027  78.640777
# 4        20  91.150442  10.442478  88.543689
# 5        25  91.150442   5.398230  94.077670
# 6        30  91.150442   2.654867  97.087379
# 7        50  91.150442   1.592920  98.252427
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/ViT-B_16/intervene_concepts_results



python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_CBM_logic" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# concepts      acc_g   acc_g_in   acc_drop
# 0         3  88.503804  83.163990   5.163324
# 1         5  88.503804  76.908707  12.231137
# 2        10  88.503804  64.792054  28.181471
# 3        15  88.503804  46.970414  47.188157
# 4        20  88.503804  30.655959  66.621777
# 5        25  88.503804  20.765850  72.796562
# 6        30  88.503804  12.411665  85.106017
# 7        50  88.503804   6.183432  93.162846
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ViT-B_16/intervene_concepts_results


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# concepts      acc_g   acc_g_in   acc_drop
# 0         3  89.687236  78.430262  11.666352
# 1         5  89.687236  68.709214  20.505184
# 2        10  89.687236  51.407439  42.256362
# 3        15  89.687236  35.472527  64.368520
# 4        20  89.687236  17.961961  80.317625
# 5        25  89.687236   7.931530  90.271442
# 6        30  89.687236   6.057481  92.475966
# 7        50  89.687236   1.714370  97.868992
(Complete)


# -----------------------------------------------------
# CUB_Resnet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/ResNet101/intervene_concepts_results
# concepts      acc_g   acc_g_in   acc_drop
# 0         3  86.538462  77.403846  10.555556
# 1         5  86.538462  74.134615  14.333333
# 2        10  86.538462  65.961538  23.777778
# 3        15  86.538462  54.807692  36.666667
# 4        20  86.538462  39.903846  53.888889
# 5        25  86.538462  30.961538  64.222222
# 6        30  86.538462  21.346154  75.333333
# 7        50  86.538462   6.538462  92.444444


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_CBM_logic" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ResNet101/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  71.198648  51.531699  22.006126
# 1         5  71.198648  42.530008  40.879020
# 2        10  71.198648  34.852071  50.977029
# 3        15  71.198648  27.244294  60.759571
# 4        20  71.198648  20.735418  70.551302
# 5        25  71.198648  13.577937  80.901991
# 6        30  71.198648   6.550296  90.568147
# 7        50  71.198648   3.55  95.162328


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline_PostHoc/ResNet101/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  80.980558  71.282164  10.482255
# 1         5  80.980558  69.329670  12.152401
# 2        10  80.980558  66.257819  17.415449
# 3        15  80.980558  55.203178  30.732777
# 4        20  80.980558  41.391378  48.356994
# 5        25  80.980558  22.233305  62.135699
# 6        30  80.980558  20.920541  73.870564
# 7        50  80.980558   2.867963  96.434238

(Complete)

# -----------------------------------------------------
# HAM10k
# -----------------------------------------------------

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 
# Acc
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/explainer/Inception_V3/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         1  93.629874  92.037342   1.700880
# 1         2  93.629874  89.676002   4.222874
# 2         3  93.629874  89.291598   4.633431
# 3         4  93.629874  89.017024   4.926686
# 4         5  93.629874  90.005491   3.870968
# 5         6  93.629874  82.646897  11.730205


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 
# Acc
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/Baseline_PostHoc/Inception_V3/intervene_concepts_results
#    concepts     acc_g   acc_g_in  acc_drop
# 0         1  91.11333  90.314528  0.876712
# 1         2  91.11333  89.066400  2.246575
# 2         3  91.11333  87.368947  4.109589
# 3         4  91.11333  83.724413  5.009589
# 4         5  91.11333  83.724413  5.59589
# 5         6  91.11333  83.724413  7.109589


# -----------------------------------------------------
# SIIM-ISIC (Canceled)
# -----------------------------------------------------

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "Inception_V3" --dataset "SIIM-ISIC" --iterations 6 --top_K 1 2 3 4 5 6 

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/SIIM-ISIC/explainer/Inception_V3/intervene_concepts_results
#    concepts      acc_g   acc_g_in  acc_drop
# 0         1  86.393089  86.177106      0.25
# 1         2  86.393089  85.529158      1.00
# 2         3  86.393089  83.801296      3.00
# 3         4  86.393089  78.617711      9.00
# 4         5  86.393089  78.185745      9.50
# 5         6  86.393089  78.185745      9.50

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "Inception_V3" --dataset "SIIM-ISIC" --iterations 6 --top_K 1 2 3 4 5 6 


# -----------------------------------------------------
# Awa2 VIT
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ViT-B_16/intervene_concepts_results
# concepts      acc_g   acc_g_in   acc_drop
# 0         3  97.572679  96.104996   1.504194
# 1         5  97.572679  93.254304   4.425803
# 2        10  97.572679  88.554897   9.242117
# 3        15  97.572679  79.904036  18.108186
# 4        20  97.572679  68.360147  29.939254
# 5        25  97.572679  51.876940  46.832514
# 6        30  97.572679  37.933954  61.122360
# 7        50  97.572679  13.886537  85.768007

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_CBM_logic" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  93.177412  91.718056   1.895093
# 1         5  93.177412  88.906052   5.024535
# 2        10  93.177412  89.803564  10.413706
# 3        15  93.177412  79.912148  16.854484
# 4        20  93.177412  66.810664  27.612521
# 5        25  93.177412  51.544094  46.368020
# 6        30  93.177412  30.035000  62.010998
# 7        50  93.177412   9.732559  90.022843

acc_g = 93
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline_PostHoc/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  96.111186  94.182184   1.966139
# 1         5  96.111186  93.067850   2.184599
# 2        10  96.111186  92.168640   3.468050
# 3        15  96.111186  85.983925  10.322228
# 4        20  96.111186  77.472873  18.016384
# 5        25  96.111186  73.730743  22.849809
# 6        30  96.111186  69.948426  27.724194
# 7        50  96.111186  23.205626  76.386128
acc_g = 96

# -----------------------------------------------------
# Awa2 Resnet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "MoIE" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  85.393428  83.629786   2.065313
# 1         5  85.393428  80.102502   6.195940
# 2        10  85.393428  72.022912  15.657546
# 3        15  85.393428  58.275550  31.756399
# 4        20  85.393428  50.331625  41.059135
# 5        25  85.393428  39.840217  53.345102
# 6        30  85.393428  29.966838  64.907326
# 7        50  85.393428  15.511004  81.835834
acc_g = 87

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_importance_intervention_main.py --model "Baseline_PCBM_logic" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline_PostHoc/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_drop
# 0         3  85.827194  65.532485  23.646012
# 1         5  85.827194  61.661085  28.156704
# 2        10  85.827194  54.748828  36.210395
# 3        15  85.827194  33.837910  60.574372
# 4        20  85.827194  21.406564  75.058530
# 5        25  85.827194  19.343603  77.462151
# 6        30  85.827194  15.083724  82.425472
# 7        50  85.827194   3.040857  96.457000

fine

CBM = 0.86



