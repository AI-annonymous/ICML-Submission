conda create -n "maskrcnn_benchmark" python=3.7 ipython
conda create -n "maskrcnn_benchmark_cuda_10" python=3.7 ipython
conda env remove -n maskrcnn_benchmark

conda install pytorch==1.0.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch-nightly
conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch 
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch

conda install pytorch==1.7.1 torchvision==0.8.2  cudatoolkit=9.0 -c pytorch-nightly
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch-nightly
conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch-nightly


https://morioh.com/p/d075922a281e
conda install -c anaconda scikit-learn
conda install -c conda-forge tensorboardx
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/BDD_OIA/maskrcnn/maskrcnn-benchmark/action_prediction/train.py --batch_size 2 --num_epoch 50 --initLR 0.001 MODEL.SIDE True MODEL.ROI_HEADS.SCORE_THRESH 0.4 MODEL.PREDICTOR_NUM 1  MODEL.META_ARCHITECTURE "Baseline1"
	

