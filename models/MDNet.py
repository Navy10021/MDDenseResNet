import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, densenet121, densenet169

################
# ResNet-based #
################
# MDResNet-18 / 34 : Optimized ResNet for Malware Detection
class MDResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=9):
        super(MDResNet18, self).__init__()
         # Load a pretrained ResNet-34 model
        resnet = resnet18(pretrained=pretrained)
        # Modify the first convolution layer to take 1 channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize it with the mean of the weights of the original first channel
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Use the remaining layers from the pretrained model
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        # Adjust the input size of the fully connected layer
        self.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output tensor
        x = self.fc(x)
        return x


class MDResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=9):
        super(MDResNet34, self).__init__()
        # Load a pretrained ResNet-34 model
        resnet = resnet34(pretrained=pretrained)

        # Modify the first convolution layer to take 1 channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize it with the mean of the weights of the original first channel
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Use the remaining layers from the pretrained model
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Adjust the fully connected layer to match the number of classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output tensor
        x = self.fc(x)
        return x


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
