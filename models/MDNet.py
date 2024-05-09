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


######################################
# DenseResNet : Residual Dense Block #
######################################
class ResidualDenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate, bn_size):
        super(ResidualDenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if input_features != growth_rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_features, growth_rate, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(growth_rate)
            )
            
    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        shortcut = self.shortcut(x)
        return torch.cat([x, out], 1)

class MDDenseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=9, growth_rate=32, bn_size=4):
        super(MDDenseResNet, self).__init__()
        self.growth_rate = growth_rate
        num_features = 64  # similar to ResNet's first conv output
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks with residual connections
        self.layers = nn.ModuleList()
        for n_blocks in num_blocks:  # Corrected iteration over num_blocks list
            layer = self._make_layer(block, num_features, n_blocks, bn_size)
            self.layers.append(layer)
            num_features += n_blocks * growth_rate
        
        self.norm_final = nn.BatchNorm2d(num_features)
        self.fc = nn.Linear(num_features, num_classes)
        
    def _make_layer(self, block, in_features, n_blocks, bn_size):
        layers = []
        for _ in range(n_blocks):
            layers.append(block(in_features, self.growth_rate, bn_size))
            in_features += self.growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pool1(self.relu(self.norm1(self.conv1(x))))
        for layer in self.layers:
            out = layer(out)
        out = F.relu(self.norm_final(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Example of creating the model
#model = DenseResNet(ResidualDenseLayer, [6, 12, 24, 16], num_classes=9)
# Light version
#model = MDDenseResNet(ResidualDenseLayer, [3, 6, 12, 8], num_classes=9, bn_size=2)
