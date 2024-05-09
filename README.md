<div align="center">

# 🛡️ MDDenseResNet: Enhanced Malware Detection Using Neural Network

</div>

---

## 📑 Project Overview

**DenseResNet** hybrid model is a conceptual framework that combines the strengths of both ResNet (Residual Network) and DenseNet (Densely Connected Network) architectures.  **MDDenseResNet** modifies the traditional **ResNet and DenseNet architecture** to enhance its capability in detecting malware, focusing on complex threats such as polymorphic and zero-day malware. This repository hosts the code, datasets, and comprehensive documentation used in our research, showcasing superior performance against traditional models like ResNet-18/34.



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

## 📦 Model 
### Key Components of DenseResNet
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
Refer to the Jupyter notebook in ***notebooks/MDResNet18_malware_500.ipynb.ipynb*** for a detailed usage example.

## 📈 Results

MDResNet models demonstrate unparalleled efficiency in identifying malicious software, particularly those elusive zero-day threats. Dive into our comprehensive performance analysis here.

***Table 1***

## 📚 Paper

- Soon!

## 👨‍💻 Contributors
- Navy Lee
