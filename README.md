# ⭐ ⭐ ⭐ DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification⭐ ⭐ ⭐ 
🛠️ This project is the PyTorch implementation of our work: "DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification".

Paper:[DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification](https://ieeexplore.ieee.org/document/10499890 "DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification")

Code:[DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification](https://github.com/RSIP-NJUPT/DPGUNet "DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification")



## 📘🛠️ Install Dependencies

* Create Conda Environment

```shell
conda create --name dpgunet python=3.10.12 -y
conda activate dpgunet
```

* Install PyTorch 1.12.1+CU116

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# if conda cannot install, use pip install (recommended and stable)
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

* Install Other Dependencies

```shell
pip install scikit-image
pip install spectral
pip install matplotlib
pip install h5py
pip install dcor
pip install opencv-python
pip install scikit-learn
pip install numpy==1.23.5
pip install mmcv-full==1.7.2
# now you need to install torch_scatter, please see https://blog.csdn.net/weixin_42421914/article/details/132875571 for details.     Note that first click "torch-1.12.1+cu116", then click "torch_scatter-2.1.0+pt112cu116-cp310-cp310-win_amd64.whl".
```
## Projection Structure <a id="1"></a>
```shell
DPGUNet
  ├── data  
  │   ├── rosenheim  
  │   ├── munich_s1  
  │   │       ├── munich_anno (gt/label)  
  │   │       ├── munich_s1 (SAR images)  
  │   │       ├── munich_segments (superpixels data)  
  │   ├── air_polsar_seg  
  ├── superpixel_hierarchy  
  │   ├── DGG.m  
  ├── config.py  
  ├── DPGUNet.py  
  ├── LICENSE  
  ├── main.py  
  ├── README.md  
  ├── utils.py  
```

👀🚀📊✨
## 📘🛠️ Prepare Dataset
### 📄Dataset introduction
- 📊**Rosenheim dataset**: The dataset is not publicly available.

- 📊**Munich dataset**: The dataset is publicly available. In fact, Munich dataset is not called the Munich dataset, but is part of ***MSLCC dataset***. You can download the dataset from the website at [MSLCC Dataset](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-51180/ "MSLCC Dataset"). If you use this dataset or view the details of the dataset, please quote or refer to the article: "[Multisensor Earth Observation Image Classification Based on a Multimodal Latent Dirichlet Allocation Model]([10.1109/LGRS.2018.2794511](https://ieeexplore.ieee.org/document/8278834) "Multisensor Earth Observation Image Classification Based on a Multimodal Latent Dirichlet Allocation Model")", thank you.

- 📊**AIR-PolSAR-Seg dataset**: The dataset is publicly available at [AIR-PolSAR-Seg Dataset](https://github.com/AICyberTeam/AIR-PolSAR-Seg "AIR-PolSAR-Seg Dataset"). For more details on the dataset see [AIR-PolSAR-Seg: A Large-Scale Data Set for Terrain Segmentation in Complex-Scene PolSAR Images](https://ieeexplore.ieee.org/document/9765389/ "AIR-PolSAR-Seg: A Large-Scale Data Set for Terrain Segmentation in Complex-Scene PolSAR Images").

### 📄Preprocessing datasets
👀 We take the Munich dataset as an example to illustrate the preprocessing process of the dataset. 

- You can download the [processed munich_s1](https://pan.baidu.com/s/1IziSwUmzf7AajIaqUUJb2w "") dataset here. (passcode: "rsip")   


**1**. **Download the [MSLCC Dataset](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-51180/ "MSLCC Dataset")**;
- Folder structure of MSLCC Dataset:
```shell
multi_sensor_landcover_classification  
├── annotations  
│   ├── berlin_anno.tif  
│   ├── ✨munich_anno.tif✨ (we used)  
├── images  
│   ├── Berlin_s1.tif (SAR image)  
│   ├── Berlin_s2.tif (multispectral image)  
│   ├── ✨Munich_s1.tif✨ (Munich SAR image we used)  
│   ├── Munich_s2.tif (multispectral image)  
├── README.txt  
```
**2**. **Crop Munich_s1.tif**
- We crop the Munich_s1.tif into patches of size 256*256 with a stride of 240. 
- 🚀**Run func "crop_munich_img()" in crop_dataset.py**.
- Put the cropped data in the munich_s1 folder.


**3**. **Crop munich_anno.tif**
- We crop the munich_anno.tif into patches of size 256*256 with a stride of 240. 
- 🚀**Run func "crop_munich_ann()" in crop_dataset.py**.
- Put the cropped data in the munich_anno folder.


**4**. **Generate hierarchy superpixels**
- 🚀Run DGG.m in dir "superpixel_hierarchy".
- Put the generated superpixels data in the munich_segments folder.


👀 At this point, we have processed the data required for training. you should put all processed data into the folder "data/munich_s1", see [Projection Structure](#Projection Structure).

## 📘🚀 Training and Testing
```shell
python main.py
```


## 🎓 Citation
  If you find our DPGUNet is useful in your research, please consider citing:
  ```shell
  @ARTICLE{10499890,
  author={Ni, Kang and Yuan, Chunyang and Zheng, Zhizhong and Huang, Nan and Wang, Peng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={DPGUNet: A Dynamic Pyramidal Graph U-Net for SAR Image Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  keywords={Convolution;Radar polarimetry;Synthetic aperture radar;Aerodynamics;Telecommunications;Land surface;Representation learning;Dynamic graph;feature fusion;graph u-net;SAR image classification;topological features},
  doi={10.1109/TAES.2024.3388373}}

  ```
