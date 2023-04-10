import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from feat.utils import euclidean_metric
from feat.networks.convnet import ConvNet
from feat.networks.resnet import ResNet
from feat.networks.amdimnet import AmdimNet


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder, hdim = self._get_encoder(args.model_type)

        self.fc = nn.Linear(hdim, args.num_class)

    def _get_encoder(self, model_type):
        if model_type == 'ConvNet':
            return ConvNet(), 64
        elif model_type == 'ResNet':
            return ResNet(), 640
        elif model_type == 'AmdimNet':
            return AmdimNet(ndf=self.args.ndf, n_rkhs=self.args.rkhs, n_depth=self.args.nd), self.args.rkhs
        else:
            raise ValueError('Invalid model type')

    def forward(self, data, is_emb=False):
        out = self.encoder(data)
        if not is_emb:
            out = self.fc(out)
        return out

    def forward_proto(self, data_shot, data_query, way=None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)

        query = self.encoder(data_query)
        logits = euclidean_metric(query, proto)
        return logits
