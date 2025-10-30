import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p: float = 0.6):
        super().__init__()
        # does all the graph maths for us...
        # we make to blocks
        self.conv1 = GCNConv(in_dim, hidden, normalize=True, cached=True)
        self.conv2 = GCNConv(hidden, hidden, normalize=True, cached=True)
        self.conv3 = GCNConv(hidden, out_dim, normalize=True, cached=True)
        # by setting some activation to zero, we can prevent overfitting... nuerons are less over-dependent on other neurons
        # also acts a way of aproximating multiple networks when we evaluate the layer multiple times...
        self.drop = nn.Dropout(p)

    # implement forward...
    def forward(self, x, edge_index):
        # x with shape: [num_nodes, in_dim]
        # edge_index are the actual connections...
        # relu to find non-linear patterns
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, normalize=True)
        self.conv2 = SAGEConv(hidden, hidden, normalize=True)
        self.conv3 = SAGEConv(hidden, out_dim, normalize=True)
        self.drop = nn.Dropout(p)

    def forward(self, x, edge_index):
        # x with shape: [num_nodes, in_dim]
        # edge_index are the actual connections...
        # relu to find non-linear patterns
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        return x
