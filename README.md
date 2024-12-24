# XMNet: XGBoost with Multitasking Network for Classification and Segmentation of Ultra-Fine-Grained Datasets

This repository contains the PyTorch implementation for the paper **"XMNet: XGBoost with Multitasking Network for Classification and Segmentation of Ultra-Fine-Grained Datasets"**.


---

## ABSTRACT
Classification and segmentation using ultra-fine-grained datasets can be challenging due to the small nuances between adjacent classes, where variations within classes can often be larger than variations between classes. To address these challenges, some approaches have employed attention mechanisms that focus on the source or properties of features causing these minor changes within or between classes. These attention mechanisms may derive from spatial, temporal, modal, or other dataset features, or even from external sources such as object shapes, skeletons, or contours. In some cases, independently extracted information is used to guide attention mechanisms in a supervised manner, such as privileged information or guided attention. In this paper, we propose that for ultra-fine datasets with a small number of samples, a simple attention mechanism can significantly improve classification results. Furthermore, this same attention mechanism can be employed in a backbone topology for segmentation, eliminating the need for additional guidance like segmentation masks, which are often used in state-of-the-art models. Unlike existing methods for ultra-fine-grained classification, such as those for plant leaves datasets that rely on segmentation masks to guide attention mechanisms, our proposed network can simultaneously provide classification labels and segmentation masks. The XGBoost algorithm is applied to the attention-modulated feature map for classification, with the Optuna hyperparameter optimization framework used to tune XGBoost. Our model, XMNet, was compared against three state-of-the-art methods on three benchmark datasets, achieving the best results for the vein segmentation task. For classification, our network delivered comparable performance to two state-of-the-art models as well as various traditional methods.



## Getting Started

### Prerequisites
To ensure a consistent development and runtime environment, we recommend using the provided `Dockerfile`. Build and run the Docker container as follows:
```bash
docker build -t xmnet .
```

### Data Preparation
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/u/2/folders/10QKsb3v__qpHuMqM96EA40M_M2DeYXN3).

Place your dataset in the `./data` directory before running the training scripts. 

---

## Training Instructions

### Step 1: Train the Classification Head
To train the classification head, use the following command:
```bash
python -u Train.py --xmnet    --cls_included   --backbone_class  $model --dataset soybean_2_1
```

For additional training options and configurations, please refer to the `train_model_cls.sh` script.

### Step 2: Train the Segmentation Head
Once the classification head is trained, train the segmentation head using the following command:
```bash
python -u Train.py --xmnet --seg_ild --freeze_all --dataparallel --data_dir ./data --backbone_class 'densenet161' --model_path best_model.pth --unet --transfer_to 0.250
```

For more segmentation training options and configurations, refer to the `train_model_seg.sh` script.

---

## Code Base
This implementation is based on [MGANet](https://github.com/Markin-Wang/MGANet). 

---


## Citation


---


