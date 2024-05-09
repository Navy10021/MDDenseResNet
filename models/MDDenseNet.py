import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, densenet169


##################
# DenseNet-based #
##################
# MDDenseNet-121 / 169 : Optimized DenseNet for Malware Detection

class MDDenseNet121(nn.Module):
    def __init__(self, pretrained=True, num_classes=9):
        super(MDDenseNet121, self).__init__()
        # Load a pretrained DenseNet-121 model
        densenet = densenet121(pretrained=pretrained)

        # Modify the first convolution layer to take 1 channel input
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Changed to 1 input channel
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0
        )

        # Append the rest of the DenseNet features
        for name, module in densenet.features.named_children():
            if name.startswith("denseblock") or name.startswith("transition"):
                self.features.add_module(name, module)

        # DenseNet final batch norm
        self.features.add_module('norm5', densenet.features.norm5)

        # Classifier head
        self.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    

class MDDenseNet169(nn.Module):
    def __init__(self, pretrained=True, num_classes=9):
        super(MDDenseNet169, self).__init__()
        # Load a pretrained DenseNet-169 model
        densenet = densenet169(pretrained=pretrained)

        # Modify the first convolution layer to take 1 channel input
        # Changing the first convolution layer in the features module
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0
        )

        # Append the rest of the DenseNet features
        for name, module in densenet.features.named_children():
            if name.startswith("denseblock") or name.startswith("transition"):
                self.features.add_module(name, module)

        # DenseNet final batch norm
        self.features.add_module('norm5', densenet.features.norm5)

        # Classifier head
        self.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
