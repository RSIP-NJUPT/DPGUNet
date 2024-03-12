# DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification
This project is the PyTorch implementation of our work: "DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification".

The complete code will be released shortly!!!

## 🛠️Step 1：Environment configuration

### Installation

```shell
$ conda create --name zoomir python=3.9
$ source activate zoomir
```

Step 1: Install PyTorch 2.0.0+CU118

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 2: Install OpenMMLab 2.x 的 `mmcv`, `mmdet`, `mmengine`, `mmsegmentation`

```shell
$ pip install -U openmim
$ mim install mmengine
$ mim install "mmcv>=2.0.0"
$ pip install "mmsegmentation>=1.0.0"
$ mim install mmdet
$ mim install "mmrotate>=1.0.0rc1"
$ mim install mmyolo
$ mim install "mmpretrain>=1.0.0rc7"
$ mim install 'mmagic'
```

Step 3: Install `zoomir`

```shell
$ git clone git@github.com:GrokCV/zoomir.git
$ cd zoomir
$ python setup.py develop
```

👀🚀📊✨
## 📘Step 2: Dataset
### 📄Dataset introduction
- ✨**Rosenheim dataset**: The dataset is not publicly available.

- ✨**Munich dataset**: The dataset is publicly available. In fact, Munich dataset is not called the Munich dataset, but is part of ***MSLCC dataset***. You can download the dataset from the website at [MSLCC Dataset](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-51180/ "MSLCC Dataset"). If you use this dataset or view the details of the dataset, please quote or refer to the article: "[Multisensor Earth Observation Image Classification Based on a Multimodal Latent Dirichlet Allocation Model]([10.1109/LGRS.2018.2794511](https://ieeexplore.ieee.org/document/8278834) "Multisensor Earth Observation Image Classification Based on a Multimodal Latent Dirichlet Allocation Model")", thank you.

- ✨**AIR-PolSAR-Seg dataset**: The dataset is publicly available at [AIR-PolSAR-Seg Dataset](https://github.com/AICyberTeam/AIR-PolSAR-Seg "AIR-PolSAR-Seg Dataset"). For more details on the dataset see [AIR-PolSAR-Seg: A Large-Scale Data Set for Terrain Segmentation in Complex-Scene PolSAR Images](https://ieeexplore.ieee.org/document/9765389/ "AIR-PolSAR-Seg: A Large-Scale Data Set for Terrain Segmentation in Complex-Scene PolSAR Images").

### 📄Preprocessing datasets
👀 We take the Munich dataset as an example to illustrate the preprocessing process of the dataset. 

1. ✨**Download the [MSLCC Dataset](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-51180/ "MSLCC Dataset")**;
- Folder structure of MSLCC Dataset:
```shell
multi_sensor_landcover_classification  
├── annotations  
│   ├── berlin_anno.tif  
│   ├── **munich_anno.tif** (we used)  
├── images  
│   ├── Berlin_s1.tif (SAR image)  
│   ├── Berlin_s2.tif (multispectral image)  
│   ├── **Munich_s1.tif** (Munich SAR image we used)  
│   ├── Munich_s2.tif (multispectral image)  
├── README.txt  
```
2. ✨**Crop Munich_s1.tif**
- We crop the Munich_s1.tif into patches of size 256*256 with a stride of 240. 
- **Run func "crop_munich_img()" in crop_dataset.py**.

3. ✨**Crop munich_anno.tif**
- We crop the munich_anno.tif into patches of size 256*256 with a stride of 240. 
- **Run func "crop_munich_ann()" in crop_dataset.py**.

4. ✨**Generate hierarchy superpixels**
- Run DGG.m in dir "superpixel_hierarchy".

👀 At this point, we have processed the data required for training.

## 📘🚀Step 3: Train