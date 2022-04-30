import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from ._base import register_model


class BaseGNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=2, dropout_p=0., **kwargs):
        super(BaseGNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList()
        self.convs.append(self.init_conv(input_size, hidden_size, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_size, hidden_size, **kwargs))
        self.convs.append(self.init_conv(hidden_size, out_size, **kwargs))

    def init_conv(self, in_size, out_size, **kwargs):
        raise NotImplementedError

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


@register_model('gcn')
class GCN(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return GCNConv(in_size, out_size, **kwargs)


@register_model('gat')
class GAT(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return GATConv(in_size, out_size, **kwargs)


@register_model('sage')
class GraphSAGE(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return SAGEConv(in_size, out_size, **kwargs)
