# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
# MoIE
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "MoIE" --epochs 3 --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108

# Epochs: 3 (Use it)
/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ViT-B_16/moIE/
   concepts     acc_bb  acc_completeness  completeness_score
0         3  92.123894         62.743363            0.302521
1         5  92.123894         74.070796            0.571429
2        10  92.123894         83.982301            0.806723
3        15  92.123894         86.283186            0.861345
4        20  92.123894         86.194690            0.869244
5        25  92.123894         86.283186            0.87345
6        30  92.123894         88.230088            0.907563

# Epochs: 100
concepts     acc_bb  acc_completeness  completeness_score
0         3  92.123894         71.681416            0.514706
1         5  92.123894         82.654867            0.775210
2        10  92.123894         88.141593            0.905462
3        15  92.123894         89.292035            0.932773
4        20  92.123894         89.115044            0.928571
5        25  92.123894         89.292035            0.932773
6        30  92.123894         89.469027            0.936975
7        50  92.123894         89.026549            0.926471


# Baseline_PCBM_logic
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "Baseline_PCBM_logic" --epochs 3 --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108
# Epochs: 3
# concepts     acc_bb  acc_completeness  completeness_score
# 0         3  91.715976         60.862215            0.230385
# 1         5  91.715976         73.034658            0.522178
# 2        10  91.715976         82.924768            0.729260
# 3        15  91.715976         84.192730            0.76656
# 4        20  91.715976         82.417582            0.777102
# 5        25  91.715976         84.615385            0.829787
# 6        30  91.715976         84.277261            0.831682

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ViT-B_16/Baseline_PostHoc
#    concepts     acc_bb  acc_completeness  completeness_score
# 0         3  91.715976         63.736264            0.329281
# 1         5  91.715976         73.964497            0.574468
# 2        10  91.715976         82.079459            0.768997
# 3        15  91.715976         84.699915            0.831814
# 4        20  91.715976         86.475063            0.874367
# 5        25  91.715976         87.066779            0.888551
# 6        30  91.715976         86.136940            0.866261
# 7        50  91.715976         86.644125            0.878419
# 8        75  91.715976         83.685545            0.807497
# 9       108  91.715976         81.318681            0.750760



# -----------------------------------------------------
# CUB_ResNet101
# -----------------------------------------------------
# MoIE

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "MoIE" --epochs 75 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108
# Epochs 75
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ResNet101/moIE
   concepts     acc_bb  acc_completeness  completeness_score
0         3  88.846154         51.634615            0.042079
1         5  88.846154         62.500000            0.321782
2        10  88.846154         74.326923            0.626238
3        15  88.846154         79.615385            0.762376
4        20  88.846154         81.634615            0.814356
5        25  88.846154         81.442308            0.809406
6        30  88.846154         82.115385            0.826733

# Epochs 25

# Baseline_PCBM_logic
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "Baseline_PCBM_logic" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "Baseline_PCBM_logic" --epochs 40 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/completeness/ResNet101/Baseline_PostHoc
#    concepts     acc_bb  acc_completeness  completeness_score
# 0         3  88.588335         53.592561            0.043100
# 1         5  88.588335         63.567202            0.301588
# 2        10  88.588335         70.836855            0.509978
# 3        15  88.588335         75.063398            0.619507
# 4        20  88.588335         78.698225            0.703702
# 5        25  88.588335         78.698225            0.723702
# 6        30  88.588335         78.444632            0.727130



# -----------------------------------------------------
# HAM10k
# -----------------------------------------------------
# MoIE

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "MoIE" --epochs 10 --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8

# Acc
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/completeness/Inception_V3/moIE
# concepts     acc_bb  acc_completeness  completeness_score
# 0         1  92.531576         86.710599            0.863138
# 1         2  92.531576         86.710599            0.863138
# 2         3  92.531576         91.543108            0.976759
# 3         4  92.531576         90.499725            0.952227
# 4         5  92.531576         93.684789            1.027114
# 5         6  92.531576         92.421746            0.997418

# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/completeness/Inception_V3/moIE
#    concepts  auroc_bb  auroc_completeness  completeness_score_auroc
# 0         1  0.973752            0.734139                  0.494223
# 1         2  0.973752            0.770542                  0.571063
# 2         3  0.973752            0.917642                  0.881562
# 3         4  0.973752            0.896945                  0.837875
# 4         5  0.973752            0.940754                  0.930347
# 5         6  0.973752            0.951408                  0.952836
# 6         7  0.973752            0.949057                  0.947873
# 7         8  0.973752            0.958938                  0.968731


# Baseline_PCBM_logic
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "Baseline_PCBM_logic" --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "Baseline_PCBM_logic" --epochs 1 --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/HAM10k/completeness/Inception_V3/Baseline_PostHoc
# concepts  auroc_bb  auroc_completeness  completeness_score_auroc
# 0         1  0.967084            0.901999                  0.860655
# 1         2  0.967084            0.925721                  0.911443
# 2         3  0.967084            0.910060                  0.877914
# 3         4  0.967084            0.923980                  0.907715
# 4         5  0.967084            0.907685                  0.872829
# 5         6  0.967084            0.922775                  0.905136
# 6         7  0.967084            0.892464                  0.840243
# 7         8  0.967084            0.923089                  0.905808


# -----------------------------------------------------
# Awa2_VIT
# -----------------------------------------------------
# MoIE
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "MoIE" --epochs 10 --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85
# Epoch 10
concepts     acc_bb  acc_completeness  completeness_score
0         3  98.334745         82.020886            0.722482
1         5  98.334745         96.401355            0.960000
2        10  98.334745         96.570703            0.963504
3        15  98.334745         96.655377            0.965255
4        20  98.334745         97.248095            0.977518
5        25  98.334745         97.417443            0.981022
6        30  98.334745         96.500141            0.962044
7        50  98.334745         96.655377            0.965255
8        75  98.334745         96.768275            0.967591
9        85  98.334745         96.613040            0.964380
# Epoch 5


# Baseline_PCBM_logic
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_concept_mask_main.py --model "Baseline_PCBM_logic" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/concept_completeness_main.py --model "Baseline_PCBM_logic" --epochs 10 --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85
# Epoch 3 (Use it)
# concepts     acc_bb  acc_completeness  completeness_score
# 0         3  98.245144         48.573342            0.008746
# 1         5  98.245144         70.381782            0.422463
# 2        10  98.245144         92.846618            0.888102
# 3        15  98.245144         96.369725            0.961127
# 4        20  98.245144         96.021433            0.953908
# 5        25  98.245144         97.628935            0.987228
# 6        30  98.245144         96.329538            0.960294
# 7        50  98.245144         97.481581            0.984173
# 8        75  98.245144         94.789015            0.928363
# 9        85  98.245144         96.275954            0.959184

# Epoch 10
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/awa2/completeness/ViT-B_16/Baseline_PostHoc
#    concepts     acc_bb  acc_completeness  completeness_score
# 0         3  98.245144         50.421969            0.008746
# 1         5  98.245144         70.247823            0.419686
# 2        10  98.245144         94.775620            0.868086
# 3        15  98.245144         97.133289            0.976954
# 4        20  98.245144         97.240455            0.979175
# 5        25  98.245144         97.709310            0.988894
# 6        30  98.245144         97.695914            0.988616
# 7        50  98.245144         97.776289            0.990282
# 8        75  98.245144         97.669123            0.988061
# 9        85  98.245144         97.441393            0.983340
