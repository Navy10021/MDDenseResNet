import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

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