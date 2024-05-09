import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50 

# MDResNet-18 / 34 : Optimized ResNet for Malware Detection
class MDResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
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
    def __init__(self, pretrained=True, num_classes=10):
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
    

# MDDenseResNet-18 / 34 : Model with improved performance using Dense Connections
"""
Dense Connections: Inspired by DenseNet,
you could modify the network to include dense connections where each layer receives additional inputs from all preceding layers,
which can improve the flow of information and gradients throughout the network.
"""
# MDResNet-18/34
class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(input_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class MDDenseResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(MDDenseResNet18, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Replace original layers with dense blocks
        self.layer1 = DenseBlock(64, num_layers=6, growth_rate=32)  # Example configuration
        self.layer2 = DenseBlock(256, num_layers=8, growth_rate=32)
        self.layer3 = DenseBlock(512, num_layers=12, growth_rate=64)
        self.layer4 = DenseBlock(1024, num_layers=6, growth_rate=64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # New fully connected layer
        self.fc = nn.Linear(1024 + 6 * 64, num_classes)  # Adjust input features according to DenseBlock output

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MDDenseResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(MDDenseResNet34, self).__init__()
        resnet = resnet34(pretrained=pretrained)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Replace original layers with dense blocks
        self.layer1 = DenseBlock(64, num_layers=6, growth_rate=32)  # Example configuration
        self.layer2 = DenseBlock(256, num_layers=8, growth_rate=32)
        self.layer3 = DenseBlock(512, num_layers=12, growth_rate=64)
        self.layer4 = DenseBlock(1024, num_layers=6, growth_rate=64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # New fully connected layer
        self.fc = nn.Linear(1024 + 6 * 64, num_classes)  # Adjust input features according to DenseBlock output

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

