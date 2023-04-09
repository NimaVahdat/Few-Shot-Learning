import torch.nn as nn

def conv_3x3(in_channels, out_channels, stride=1):
    """Create a 3x3 convolutional layer with specified in/out channels and stride."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block with two 3x3 convolutional layers."""
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block with 1x1, 3x3, and 1x1 convolutional layers."""
    
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import torch.nn as nn

class ResNet(nn.Module):
    """
    A ResNet-based neural network architecture.
    """
    def __init__(self, block=BasicBlock, layers=[4,4,4]):
        """
        Initializes the ResNet architecture.
        
        Args:
        - block: the type of residual block to use (default: BasicBlock)
        - layers: a list indicating the number of blocks for each layer (default: [4,4,4])
        """
        super(ResNet, self).__init__()
        cfg = [160, 320, 640] # channel configuration
        self.inplanes = iChannels = int(cfg[0]/2) # number of input channels
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1) # first convolutional layer
        self.bn1 = nn.BatchNorm2d(iChannels) # batch normalization
        self.relu = nn.ReLU(inplace=True) # activation function
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2) # first residual layer
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2) # second residual layer
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2) # third residual layer
        self.avgpool = nn.AvgPool2d(10, stride=1) # average pooling layer

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer.
        
        Args:
        - block: the type of residual block to use
        - planes: the number of output channels
        - blocks: the number of blocks to use
        - stride: the stride of the first block (default: 1)
        
        Returns:
        A sequential module representing the residual layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Computes a forward pass through the network.
        
        Args:
        - x: the input tensor
        
        Returns:
        The output tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # no fully connected layer here
        return x
