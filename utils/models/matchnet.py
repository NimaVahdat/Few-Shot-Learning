import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BidirectionalLSTM(nn.Module):
    """
    A multi-layer bidirectional LSTM implementation for embedding adaptation.
    """
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM.

        layer_sizes: A list containing the neuron numbers per layer.
        batch_size: The experiments batch size.
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        
        inputs: The inputs should be a list of shape [sequence_length, batch_size, 64].
        return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        if torch.cuda.is_available():
            c0 = c0.cuda()
            h0 = h0.cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn
    

class MatchNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_bilstm = args.use_bilstm
        self.args = args  # information about Shot and Way

        model_types = {
            'ConvNet': ('feat.networks.convnet', 'ConvNet', 32),
            'ResNet': ('feat.networks.resnet', 'ResNet', 320),
            'AmdimNet': ('feat.networks.amdimnet', 'AmdimNet', None),
        }

        module_name, class_name, layer_size = model_types.get(args.model_type, (None, None, None))
        if not module_name:
            raise ValueError('Unknown model type')

        module = __import__(module_name, fromlist=[class_name])
        encoder_class = getattr(module, class_name)
        self.encoder = encoder_class(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd) if class_name == 'AmdimNet' else encoder_class()
        if not layer_size:
            layer_size = int(args.rkhs / 2)

        if self.use_bilstm:
            self.bilstm = BidirectionalLSTM(layer_sizes=[layer_size], batch_size=args.query * args.way, vector_dim=layer_size * 2)

    def forward(self, support_set, query_set):
        # produce embeddings for support set images
        support_set = self.encoder(support_set)  # KN x d
        # produce embedding for target images
        query_set = self.encoder(query_set)  # KqN x d

        num_support = support_set.shape[0]
        num_query = query_set.shape[0]
        support_extend = support_set.unsqueeze(0).repeat([num_query, 1, 1])  # KqN x KN x d
        query_extend = query_set.unsqueeze(1)  # KqN x 1 x d
        combined = torch.cat([support_extend, query_extend], 1)  # KqN x (KN + 1) x d

        if self.use_bilstm:
            # FCE embedding
            combined = combined.permute([1, 0, 2])  # (KN + 1) x KqN x d
            combined, hn, cn = self.bilstm(combined)
            combined = combined.permute([1, 0, 2])  # KqN x (KN + 1) x d

        # get similarity between support set embeddings and target
        refined_support, refined_query = combined.split((self.args.shot * self.args.way), 1)  # KqN x

        # compute cos similarity
        refined_support = F.normalize(refined_support, dim=2)  # KqN x KN x d
        # compute inner product, batch inner product
        logits = torch.bmm(refined_support, refined_query.permute([0, 2, 1])) / self.args.temperature  # KqN x KN x d * KqN x d x 1

        return logits  # KqN x KN x 1