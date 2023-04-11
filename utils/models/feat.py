import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from feat.utils import euclidean_metric
from scipy.io import loadmat

class ScaledDotProductAttention(nn.Module):
    # Scaled Dot-Product Attention
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    # Multi-Head Attention module 
    def __init__(self, args, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = nn.Linear(d_model, n_head * d_k)(q).view(sz_b, len_q, n_head, d_k)
        k = nn.Linear(d_model, n_head * d_k)(k).view(sz_b, len_k, n_head, d_k)
        v = nn.Linear(d_model, n_head * d_v)(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(nn.Module):
    def __init__(self, args, dropout=0.2):
        """
        Initializes the FEAT module with the given arguments.

        Args:
            args: A Namespace object with the arguments for the module.
            dropout: The dropout probability for the multi-head attention layer.
        """
        super().__init__()
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
            self.z_dim = 64
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
            self.z_dim = 640
        elif args.model_type == 'AmdimNet':
            from feat.networks.amdimnet import AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
            self.z_dim = args.rkhs
        else:
            raise ValueError('Invalid model type')

        self.slf_attn = MultiHeadAttention(args, args.head, self.z_dim, self.z_dim, self.z_dim, dropout=dropout)
        self.args = args

    def forward(self, support, query, mode='test'):
        """
        Runs the forward pass of the FEAT module with the given inputs.

        Args:
            support: A tensor with shape (B, C, H, W) representing the support set.
            query: A tensor with shape (B, C, H, W) representing the query set.
            mode: A string specifying the mode of the forward pass, either 'train' or 'test'.

        Returns:
            A tensor with shape (B, N) representing the logits for the query set.
        """
        # Feature extraction
        support = self.encoder(support)  # Shape: (B, z_dim, 1, 1)
        # Get mean of the support
        proto = support.reshape(self.args.shot, -1, self.z_dim).mean(dim=0)  # Shape: (N, z_dim)
        num_proto = proto.shape[0]
        # For query set
        query = self.encoder(query)

        # Adapt the support set instances
        proto = proto.unsqueeze(0)  # Shape: (1, N, z_dim)
        # Refine by multi-head attention
        proto = self.slf_attn(proto, proto, proto)
        proto = proto.squeeze(0)

        # Compute distance for all batches
        logits = euclidean_metric(query, proto) / self.args.temperature

        if mode == 'train':
            # Transform for all instances in the task
            aux_task = torch.cat([support.reshape(self.args.shot, -1, self.z_dim),
                                  query.reshape(self.args.query, -1, self.z_dim)], 0)  # Shape: ((K + Kq), N, z_dim)
            aux_task = aux_task.permute([1, 0, 2])
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task)  # Shape: (N, (K + Kq), z_dim)
            # Compute class mean
            aux_center = torch.mean(aux_emb, 1)  # Shape: (N, z_dim)
            logits2 = euclidean_metric(aux_task.permute([1, 0, 2]).view(-1, self.z_dim), aux_center) / self.args.temperature2
            return logits, logits2
        else:
            return logits
