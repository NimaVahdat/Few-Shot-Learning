import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ConvNet(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        super(ConvNet, self).__init__()
        self.block1 = ConvBlock(input_dim, hidden_dim)
        self.block2 = ConvBlock(hidden_dim, hidden_dim)
        self.block3 = ConvBlock(hidden_dim, hidden_dim)
        self.block4 = ConvBlock(hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.MaxPool2d(kernel_size=5)(x)
        x = x.view(x.size(0), -1)
        return x
