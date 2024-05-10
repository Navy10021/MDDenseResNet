<div align="center">

# 🛡️ MDDenseResNet: Enhanced Malware Detection Using Deep Neural Network

</div>

---

## 📑 Project Overview
 With the advent of the digital age, cyberspace has been exposed to a variety of threats. Among these, malware represents a significant and potent cyber threat. Operating without user consent, it can steal information, damage systems, and initiate widespread network attacks. A systematic approach to prevention and response is essential to combat the threats posed by malware. 

 In this project, we transform malware into images and introduce an advanced neuron network based malware detection AI model. Inspired by **ResNet (Microsoft., 2015)** and **DenseNet (Meta., 2016)** image analysis models, we develop 1️⃣ ***MDResNet(Residual Network for Malware Detection)***  2️⃣ ***MDDenseNet(Dense Network Malware Detection)*** and a hybrid model, 3️⃣ ***MDDenseResNet***.

## 📋 Table of Contents

- [Dataset](#Dataset)
- [Usage](#Usage)
- [Results](#Results)
- [License](#License)

## 📊 Dataset

Our models are trained on diverse malware image datasets, designed to challenge and test the robustness of MDResNet:

- **Malware-500**: Derived from 12,000 base samples across 9 classes, using undersampling to ensure equity.
- **Malware-1000**: 10,000 images, enhanced with data augmentation for richer training data.
- **Malware-5000**: Our largest dataset with 50,000 images, providing the depth needed for comprehensive model training.

Refer to the Jupyter notebook in ***notebooks/Malware_Dataset.ipynb***

## 📦 Model 
### Key Components of MDDenseResNet
### 1. Residual Learning (from ResNet)
- **Residual Blocks**: These are the building blocks of ResNet, where the input to a block is directly added to the output of the block through a shortcut connection. This helps in training deeper networks by enabling direct backward paths for gradients during training, effectively addressing the vanishing gradient problem.
### 2. Dense Connectivity (from DenseNet)
- **Dense Blocks**: Each layer within a dense block receives feature maps from all preceding layers as input and passes its own feature maps to all subsequent layers. This ensures maximum information flow between layers in the network, leading to highly efficient use of features and reduced redundancy in feature learning.

### Benefits of DenseResNet
- **Enhanced Gradient Flow**: Combines the direct gradient paths of ResNet with the enriched feature reuse of DenseNet, potentially leading to better training dynamics and convergence.
- **Improved Feature Reuse**: Enables the model to leverage a wide range of features from both simple and complex representations, which is particularly useful for tasks involving fine-grained recognition or detailed textural information.
- **Resource Efficiency**: By combining these approaches, DenseResNet can achieve comparable or superior performance with fewer parameters and less computational overhead compared to using either architecture alone.

## 🚀 Usage and Example
Data preprocessing and Train & evaluate the models with ease :
Refer to the Jupyter notebook in ***notebooks/MDDenseResNet18_malware_500.ipynb.ipynb*** for a detailed usage example.

## 📈 Results

MDResNet models demonstrate unparalleled efficiency in identifying malicious software, particularly those elusive zero-day threats. Dive into our comprehensive performance analysis here.

***Table 1***

## 📚 Paper

- Soon!

## 👨‍💻 Contributors
Seoul National University GSDS NLP Labs
Under the guidance of ***Navy Lee***
