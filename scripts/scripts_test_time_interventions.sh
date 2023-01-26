# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in  acc_gain
# 0         3  91.150442  93.628319  2.718447
# 1         5  91.150442  95.132743  4.368932
# 2        10  91.150442  96.283186  5.631068
# 3        15  91.150442  97.433628  6.893204
# 4        20  91.150442  98.495575  8.058252
# 5        25  91.150442  98.318584  7.864078
# 6        30  91.150442  98.849558  8.446602
# 7        50  91.150442  99.380531  9.029126

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline/ViT-B_16/intervene_concepts_results
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_CBM_logic" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# concepts      acc_g   acc_g_in   acc_gain
# 0         3  88.334742  91.054100   4.210526
# 1         5  88.334742  92.575655   5.933014
# 2        10  88.334742  94.435334   8.038278
# 3        15  88.334742  95.196112   8.899522
# 4        20  88.334742  96.379544  10.239234
# 5        25  88.334742  96.717667  10.622010
# 6        30  88.334742  97.140321  11.100478
# 7        50  88.334742  98.154691  12.248804

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/Baseline_PostHoc/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_gain
# 0         3  89.687236  91.068470   3.770028
# 1         5  89.687236  92.251902   5.089538
# 2        10  89.687236  93.688926   6.691800
# 3        15  89.687236  94.125951   8.294062
# 4        20  89.687236  95.886729   9.142319
# 5        25  89.687236  96.478445   9.802074
# 6        30  89.687236  96.647506   9.990575
# 7        50  89.687236  97.239222  10.650330


# -----------------------------------------------------
# CUB_ResNet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/ResNet101/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_gain
# 0         3  86.538462  87.980769   1.666667
# 1         5  86.538462  88.557692   2.333333
# 2        10  86.538462  90.769231   4.888889
# 3        15  86.538462  93.173077   7.666667
# 4        20  86.538462  93.846154   8.444444
# 5        25  86.538462  95.576923  10.444444
# 6        30  86.538462  95.769231  10.666667
# 7        50  86.538462  97.211538  12.333333

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_CBM_logic" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
#  concepts      acc_g   acc_g_in   acc_gain
# 0         3  71.029586  73.213018   2.150538
# 1         5  55.029586  75.495351   6.298003
# 2        10  55.029586  76.340659   7.834101
# 3        15  55.029586  78.608622  10.138249
# 4        20  55.029586  80.777684  10.445469
# 5        25  55.029586  79.847844   8.755760
# 6        30  55.029586  81.115807  11.059908
# 7        50  55.029586  81.608622  10.138249

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_PCBM_logic" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50
#  concepts      acc_g   acc_g_in   acc_gain
# 0         3  80.980558  83.685545   3.340292
# 1         5  80.980558  84.277261   4.070981
# 2        10  80.980558  86.390533   6.680585
# 3        15  80.980558  87.743026   8.350731
# 4        20  80.980558  89.940828  11.064718
# 5        25  80.980558  91.377853  12.839248
# 6        30  80.980558  92.983939  14.822547
# 7        50  80.980558  95.773457  18.267223


# -----------------------------------------------------
# Awa2_vit
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 

# concepts      acc_g   acc_g_in  acc_gain
# 0         3  97.572679  97.854925  0.289268
# 1         5  97.572679  98.024273  0.462829
# 2        10  97.572679  98.772227  1.229390
# 3        15  97.572679  99.167372  1.634365
# 4        20  97.572679  99.237934  1.706682
# 5        25  97.572679  99.294383  1.764536
# 6        30  97.572679  99.280271  1.750072
# 7        50  97.572679  99.393170  1.865780
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_CBM_logic" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline/ViT-B_16/intervene_concepts_results
#    concepts      acc_g   acc_g_in  acc_gain
# 0         3  93.182853  93.557937  0.473693
# 1         5  79.182853  93.799062  0.778210
# 2        10  79.182853  93.866042  0.862798
# 3        15  79.182853  93.973208  0.998139
# 4        20  79.182853  94.013396  1.048892
# 5        25  79.182853  94.013396  1.048892
# 6        30  79.182853  95.040188  1.082727
# 7        50  79.182853  95.174146  1.251903

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline_PostHoc/ViT-B_16/intervene_concepts_results
#    concepts      acc_g    acc_g_in  acc_gain
# 0         3  96.111186   96.648895  0.955762
# 1         5  98.111186   97.410583  1.324413
# 2        10  98.111186   97.705291  1.624795
# 3        15  98.111186   97.892833  1.815948
# 4        20  98.111186   97.933021  1.856909
# 5        25  98.111186   98.933021  1.856909
# 6        30  98.111186   99.959812  1.884216
# 7        50  98.111186  100.000000  1.925177

# -----------------------------------------------------
# Awa2_ResNet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "MoIE" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ResNet50/intervene_concepts_results
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/explainer/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_gain
# 0         3  87.393428  91.478444   4.783760
# 1         5  85.393428  93.211938   6.813769
# 2        10  85.393428  95.442870   9.426302
# 3        15  85.393428  96.146216  11.421006
# 4        20  85.393428  97.764245  12.144748
# 5        25  85.393428  98.061682  12.727273
# 6        30  85.393428  98.894784  13.468667
# 7        50  85.393428  99.270546  14.845543

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_CBM_logic" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_gain
# 0         3  86.737441  87.969859   1.933586
# 1         5  63.737441  88.947756   3.467844
# 2        10  63.737441  90.081112   5.716688
# 3        15  63.737441  92.894843   8.091635
# 4        20  63.737441  94.165573   9.457755
# 5        25  63.737441  95.542532  10.676755
# 6        30  63.737441  96.480241  12.147961
# 7        50  63.737441  98.052244  16.183270

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --model "Baseline_PCBM_logic" --arch "ResNet50" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/Baseline_PostHoc/ResNet50/intervene_concepts_results
#    concepts      acc_g   acc_g_in   acc_gain
# 0         3  85.827194  89.537843   4.323396
# 1         5  85.827194  90.341594   5.259872
# 2        10  85.827194  93.020764   8.381458
# 3        15  85.827194  94.534494  10.145154
# 4        20  85.827194  95.432016  11.190885
# 5        25  85.827194  96.128600  12.002497
# 6        30  85.827194  96.342934  12.252224
# 7        50  85.827194  97.374414  13.454035

