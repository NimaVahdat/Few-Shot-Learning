import torch.nn as nn

class ProtoNet(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = self.get_encoder(args.model_type, args)

    def forward(self, data_shot, data_query):
        proto = self.get_prototype(data_shot)
        logits = self.get_logits(data_query, proto)
        return logits

    def get_encoder(self, model_type, args):
        if model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            return ConvNet()
        elif model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            return ResNet()
        elif model_type == 'AmdimNet':
            from feat.networks.amdimnet import AmdimNet
            return AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
        else:
            raise ValueError('Invalid model type.')

    def get_prototype(self, data_shot):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        return proto
    
    def get_logits(self, data_query, proto):
        from feat.utils import euclidean_metric
        logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
        return logits
