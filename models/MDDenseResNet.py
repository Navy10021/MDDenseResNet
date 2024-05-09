"""
DenseResNet : Hybride model combined ResNet and DenseNet
Combining ResNet and DenseNet architectures can yield a model that leverages the strengths of both: the residual learning features of ResNet and the feature reuse and dense connectivity of DenseNet. 
This combination can be particularly useful for ensuring robust and efficient feature propagation across the network, which is especially beneficial in deeper networks. 

1.Residual Dense Block: Combine the concepts of residual connections and dense connectivity within each block.
Each layer in a dense block not only connects to all preceding layers but also includes a residual connection that bypasses the dense connections directly to the output of the block.

2.Integrated Transition Layers: Use ResNet-style bottleneck layers as transition layers within the DenseNet architecture to reduce dimensionality and to help in regularizing the model.

3.Hybrid Feature Extraction: Start with a few ResNet-style convolutional and bottleneck layers for initial feature extraction, which reduces the spatial dimensions while increasing the depth.
Then, continue with DenseNet-style dense blocks for detailed feature extraction and increased reuse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
model = MDDenseResNet(ResidualDenseLayer, [3, 6, 12, 8], num_classes=9, bn_size=2)
