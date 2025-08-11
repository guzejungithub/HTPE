# Human-Structure-Aware Token Position Embedding for Tokenized Pose Estimation
Official Implementation for: Human-Structure-Aware Token Position Embedding for Tokenized Pose Estimation

> [**Human-Structure-Aware Token Position Embedding for Tokenized Pose Estimation**],            



# Installation & Quick Start
HTPE referenced <a href="https://github.com/yshMars/DistilPose">DistilPose (CVPR 2023) </a> and is developed using MMPose and Pytorch framework. Please install the relevant packages listed below:
```
conda create -n htpe python=3.8 pytorch=1.7.0 torchvision -c pytorch -y
conda activate htpe
pip3 install openmim
mim install mmcv-full==1.3.8
git submodule update --init
cd mmpose
git checkout v0.29.0
pip3 install -e .
cd ..
pip3 install -r requirements.txt
```
For training on COCO, you will need to download the official COCO dataset and modify the dataset path in the model configuration files. After these setups, run the following command lines:
```
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/htpe/coco/htpe_s_v1_stemnet_coco_256x192.py 8
```
For evaluating on COCO, downlowd checkpoint and run the following command lines:
```
./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/distilpose/coco/htpe_s_v1_stemnet_coco_256x192.py \
./checkpoints/htpe_s_v1.pth 8
```


```
