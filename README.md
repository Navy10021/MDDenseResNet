<div align="center">

# 🛡️ MDDenseResNet: Enhanced Malware Detection Using Deep Neural Networks

</div>

---

## 📑 Project Overview
With the advent of the digital age, cyberspace has been exposed to a variety of threats. Among these, malware represents a significant and potent cyber threat. Operating without user consent, it can steal information, damage systems, and initiate widespread network attacks. A systematic approach to prevention and response is essential to combat the threats posed by malware.

In this project, we transform malware into images and introduce an advanced neuron network-based malware detection AI model. Inspired by **ResNet (Microsoft, 2015)** and **DenseNet (Meta, 2016)** image analysis models, we develop three models:
1. ***MDResNet*** (Residual Network for Malware Detection)
2. ***MDDenseNet*** (Dense Network for Malware Detection)
3. ***MDDenseResNet*** (Hybrid Model for Malware Detection)

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Models](#-models)
- [Usage](#-usage-and-example)
- [Results](#-results)
- [License](#-license)
- [Paper](#-paper)
- [Contributors](#-contributors)

## 📊 Dataset

Our models are trained on diverse malware image datasets, designed to challenge and test the robustness of MDDenseResNet:

- **MalwarePix-small**: 3,915 images, derived from 12,000 base samples across 9 classes, using undersampling to ensure equity.
- **MalwarePix-medium**: 13,254 images, enhanced with data augmentation for richer training data.
- **MalwarePix-large**: Our largest dataset with 26,478 images, providing the depth needed for comprehensive model training.

Refer to the Jupyter notebook in [notebooks/Malware_Dataset.ipynb](notebooks/Malware_Dataset.ipynb) for more details.

**You can download the data from the following Google Drive folder:**
📁 [Download Data from Google Drive](https://drive.google.com/drive/folders/1d6pnDUoJt7tDYyCFxDcDiTYa-2TdVn7N)
  
![image](https://github.com/Navy10021/MDDenseResNet/assets/105137667/6c6d3ab7-6f68-4773-b8a6-6937de17e503)


## 📦 Models: MDResNet | MDDenseNet | MDDenseResNet
### Key Components of MDDenseResNet
#### 1. Residual Learning (from ResNet)
- **Residual Blocks**: The building blocks of ResNet, where the input to a block is directly added to the output of the block through a shortcut connection. This helps in training deeper networks by enabling direct backward paths for gradients during training, effectively addressing the vanishing gradient problem.

#### 2. Dense Connectivity (from DenseNet)
- **Dense Blocks**: Each layer within a dense block receives feature maps from all preceding layers as input and passes its own feature maps to all subsequent layers. This ensures maximum information flow between layers in the network, leading to highly efficient use of features and reduced redundancy in feature learning.

### Benefits of MDDenseResNet
- **Enhanced Gradient Flow**: Combines the direct gradient paths of ResNet with the enriched feature reuse of DenseNet, potentially leading to better training dynamics and convergence.
- **Improved Feature Reuse**: Enables the model to leverage a wide range of features from both simple and complex representations, which is particularly useful for tasks involving fine-grained recognition or detailed textural information.
- **Resource Efficiency**: By combining these approaches, MDDenseResNet can achieve comparable or superior performance with fewer parameters and less computational overhead compared to using either architecture alone.

## 🚀 Usage and Example

### Overall Pipeline

![image](https://github.com/Navy10021/MDDenseResNet/assets/105137667/f4ee0df5-6893-45e6-aa85-8929c2f55c6a)


### Usage
Data preprocessing and training & evaluation of the models with ease:
```bash
$ python train/mdresnet18.py
$ python train/mddensenet121.py
$ python train/mddenseresnet.py
```
Refer to the Jupyter notebook in [notebooks/MDResNet18_malware_medium.ipynb](notebooks/MDResNet18_malware_medium.ipynb) for a detailed usage example.

## 📈 Results

MDResNet models demonstrate unparalleled efficiency in identifying malicious software, particularly those elusive zero-day threats. Dive into our comprehensive performance analysis here.

![Table1](https://github.com/user-attachments/assets/407f9f8b-c590-419b-80ec-49bad75f779d)

## 📚 Paper

- 📝 ***심층 신경망 기반 차세대 악성코드 탐지에 관한 연구: 악성코드 시각화 및 고성능 분류 / 이상 탐지 모델 개발***
- 📝 ***Deep Neural Networks for Next-Generation Malware Detection: Malware Visualization Techniques and High-Performance Classification / Anomaly Detection Models***

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

