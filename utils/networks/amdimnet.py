import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Flattens the tensor
def flatten_tensor(x):
    return x.reshape(x.size(0), -1)

# Module to flatten the tensor
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

class AmdimNet(nn.Module):
    def __init__(self, num_channels=3, ndf=256, n_rkhs=2048, n_depth=10, encoder_size=128, use_bn=False):
        # call the constructor of the parent class
        super().__init__()

        # store the values of the parameters as class attributes
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        # create a dummy batch of input data to configure the network
        dummy_batch = torch.zeros((2, num_channels, encoder_size, encoder_size))

        # create the encoding block for local features
        self.layer_list = nn.ModuleList([
            Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
            Conv3x3(ndf, ndf, 3, 1, 0, False),
            ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
            ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
            ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
            MaybeBatchNorm2d(ndf * 8, True, use_bn),
            ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
            ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
            ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
            MaybeBatchNorm2d(n_rkhs, True, True)
        ])

        # configure the network for extracting fake rkhs embeddings for infomax
        self._config_modules(dummy_batch, [1, 5, 7], n_rkhs, use_bn)

    def _config_modules(self, x, rkhs_layers, n_rkhs, use_bn):
        # get activations at each layer of the network for the dummy input data
        enc_acts = self._forward_acts(x)

        # create a dictionary that maps layer dimensions to layer indices
        self.dim2layer = {}
        for i, h_i in enumerate(enc_acts):
            for d in rkhs_layers:
                if h_i.size(2) == d:
                    self.dim2layer[d] = i

        # get the feature sizes at different layers
        self.ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        self.ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        self.ndf_7 = enc_acts[self.dim2layer[7]].size(1)

        # create the modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(self.ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(self.ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        # compute activations in all layers for x
        acts = self._forward_acts(x)
        # gather rkhs embeddings from certain layers
        r1 = self.rkhs_block_1(acts[self.dim2layer[1]])
        #r5 = self.rkhs_block_5(acts[self.dim2layer[5]])
        #r7 = self.rkhs_block_7(acts[self.dim2layer[7]])
        r1 = r1.view(r1.size(0), -1)
        return r1


class MaybeBatchNorm2d(nn.Module):
    def __init__(self, n_ftr, affine, use_bn):
        super().__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        else:
            self.bn = None

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x



class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super().__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = x.norm(dim=self.norm_dim, keepdim=True)
            x = x / (x_norms + 1e-6)
        return x



class Conv3x3(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad,
                 use_bn=True, pad_mode='constant'):
        super(Conv3x3, self).__init__()
        assert(pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0,
                              bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn) if use_bn else None

    def forward(self, x):
        if self.n_pad[0] > 0:
            # pad the input if required
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # conv is always applied
        x = self.conv(x)
        # apply batchnorm if required
        if self.bn is not None:
            x = self.bn(x)
        # relu is always applied
        out = self.relu(x)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = nn.Dropout(p=p)
        self.linear1 = nn.Linear(n_input, n_hidden, bias=False) if n_hidden is not None else None
        self.bn1 = nn.BatchNorm1d(n_hidden) if n_hidden is not None else None
        self.relu = nn.ReLU(inplace=True) if n_hidden is not None else None
        self.linear2 = nn.Linear(n_hidden or n_input, n_classes, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        if self.n_hidden is not None:
            x = self.linear1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
        logits = self.linear2(x)
        return logits


class FakeRKHSConvNet(nn.Module):
    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn_hid = MaybeBatchNorm2d(n_output, True, use_bn)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def init_weights(self, init_scale=1.):
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        nn.init.constant_(self.conv2.weight, 0.)

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn_hid(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h



class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=False):
        super(ConvResBlock, self).__init__()
        
        assert (out_channels >= in_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.conv3 = None
        
        self.batch_norm = MaybeBatchNorm2d(out_channels, True, use_bn)
        
    def init_weights(self, init_scale=1.):
        # initialize the weights of the first convolution layer
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        
        # initialize the weights of the second convolution layer to zero
        nn.init.constant_(self.conv2.weight, 0.)
        
    def forward(self, x):
        # apply batch normalization and the first convolution layer
        h1 = self.batch_norm(self.conv1(x))
        
        # apply the second convolution layer and ReLU activation function
        h2 = self.conv2(self.relu2(h1))
        
        # apply the shortcut connection
        if (self.out_channels < self.in_channels):
            h3 = self.conv3(x)
        elif (self.in_channels == self.out_channels):
            h3 = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        else:
            h3_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.out_channels - self.in_channels))
        
        # add the results from the second convolution layer and shortcut connection
        h23 = h2 + h3
        
        return h23


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depth, use_batchnorm):
        super(ConvResidualBlock, self).__init__()

        # Create list of ConvResNxN layers with given depth
        layers = [ConvResNxN(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm)]
        for i in range(depth - 1):
            layers.append(ConvResNxN(out_channels, out_channels, 1, 1, 0, use_batchnorm))

        # Use nn.Sequential to create a module with the list of layers
        self.layer_list = nn.Sequential(*layers)

    def init_weights(self, init_scale=1.):
        '''
        Initialize the weights of each ConvResNxN layer in this block using fixup-style initialization.
        '''
        for layer in self.layer_list:
            layer.init_weights(init_scale)

    def forward(self, x):
        # Pass input through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out

