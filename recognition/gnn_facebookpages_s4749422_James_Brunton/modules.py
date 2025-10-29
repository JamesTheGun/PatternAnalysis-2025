import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, normalize=True, cached=True)
        self.conv2 = GCNConv(hidden, out_dim, normalize=True, cached=True)
        self.drop = nn.Dropout(p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x


class graphSampleAndAgrigate(GraphConvolutionalNetwork):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, normalize=True)
        self.conv2 = SAGEConv(hidden, out_dim, normalize=True)
        self.drop  = nn.Dropout(p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x
