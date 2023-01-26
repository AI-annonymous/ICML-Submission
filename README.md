# ICML annonymous

## Environment setup

Use the file **environment.yml** to create the environment.

## Data Instructions

Below we list the data sources to download the datasets we used to train / evaluate MoIEs. Once you download a dataset,
update the correct paths in the variable `--data-root` for the `[files]_main.py` files.

## Downloading the Data

| Dataset | Description                        | URL
|---------|------------------------------------|--------------------------------------------------------------------------------------------------------
| CUB-200 | Bird Classification dataset        | [CUB-200 Official](https://www.vision.caltech.edu/datasets/cub_200_2011/)                              |
| CUB-200 | CUB Metadata and splits            | [Logic Explained network](https://github.com/pietrobarbiero/logic_explained_networks/tree/master/data) |
| Derm7pt | Dermatology Concepts Dataset       | [Get access here](https://derm.cs.sfu.ca/Welcome.html)                                                 |
| HAM10k  | Skin lesion classification dataset | [Kaggle Link](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)                                |
| Awa2    | Animals with Attributes2           | [Awa2 official](https://cvml.ista.ac.at/AwA2/)                                                         |

## Running the pipepline

* Go the **scripts** folder and get the training scripts. For all the datasets, one script is included with proper
  instructions to run 1) Blackbox 2) projection (t) 3) interpretable model (g) 4) residual (r). **For example, to run
  CUB-200 for ResNet101, look into the file `scripts/cub_resnet.py`**
* The naming convention and the paths to be replaced is mentioned in the script. Follow them carefully
* Run them sequentially.
* Also, we have added the code to run the BlackBox for Awa2 and CUB-200. For HAM10k and ISIC please go to the Post-hoc
  concept Bottleneck ([PCBM](https://github.com/mertyg/post-hoc-cbm)) repo and get the pretrained Blackbox and the
  concepts.
* Due to anonymity, we can not upload the pretrained models. Upon decision we will upload the pretrained model as well.

# References
* [ResNet-101 on CUB-200](https://github.com/zhangyongshun/resnet_finetune_cub)
* [VIT-B_16 on CUB-200] (https://github.com/TACJu/TransFG)
* [Models and concepts for HAM10k and ISIC] (https://github.com/mertyg/post-hoc-cbm)
* [FOL] (https://github.com/pietrobarbiero/entropy-lens)
* [Concept Bottleneck Models] (https://github.com/yewsiang/ConceptBottleneck)
