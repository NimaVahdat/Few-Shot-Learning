import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn, log_attn


class FEAT(nn.Module):
    def __init__(self, args, dropout=0.2):
        super().__init__()
        self.args = args
        self._set_encoder(args)
        self.slf_attn = MultiHeadAttention(1, self.z_dim, self.z_dim, self.z_dim, dropout=dropout)
    
    def forward(self, support, query):
        support, proto = self._extract_features(support)
        query = self.encoder(query)
        combined = self._combine_features(proto, query)
        combined, enc_slf_attn, enc_slf_log_attn = self.slf_attn(combined, combined, combined)
        refined_support, refined_query = combined.split(self.args.way, 1)
        logitis = -torch.sum((refined_support - refined_query) ** 2, 2) / self.args.temperature
        return logitis, enc_slf_log_attn
    
    def _set_encoder(self, args):
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
            self.z_dim = 64
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
            self.z_dim = 640
        else:
            raise ValueError('')
    
    def _extract_features(self, x):
        x = self.encoder(x)
        proto = x.reshape(self.args.shot, -1, x.shape[-1]).mean(dim=0)
        return x, proto
    
    def _combine_features(self, proto, query):
        num_query = query.shape[0]
        proto = proto.unsqueeze(0).repeat([num_query, 1, 1])
        query = query.unsqueeze(1)
        combined = torch.cat([proto, query], 1)
        return combined
