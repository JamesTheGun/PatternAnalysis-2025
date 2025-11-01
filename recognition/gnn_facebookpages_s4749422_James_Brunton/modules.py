import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, Sequential


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


class blockGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        block_layer_count=4,
        block_layer_size=64,
        block_count=5,
        p=0.75,
    ):
        super().__init__()
        self.block_layer_count = block_layer_count
        self.block_layer_size = block_layer_size
        self.block_count = block_count
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.drop = nn.Dropout(p)
        self._construct_shit()

    def _construct_shit(self):
        self._make_conv_in()
        self._make_conv_out()
        self._make_blocks()

    class block(nn.Module):
        def __init__(
            self,
            block_layer_count=5,
            layer_size=64,
            p=0.75,
        ):
            super().__init__()
            self.p = p
            self.layers = nn.ModuleList(
                [
                    GCNConv(layer_size, layer_size, normalize=True, cached=True)
                    for i in range(block_layer_count)
                ]
            )
            self.drop = nn.Dropout(p)

        def forward(self, x, edge_index):
            # arn't these rediduals buetiful?? like it should not be this simple
            # also torch tensors don't need copy?? seems like vudoo
            x_in = x
            for layer in self.layers:
                x = layer(x, edge_index).relu()
                x = self.drop(x)
            x = x + x_in
            return x

    def _make_blocks(self):
        self.blocks = nn.ModuleList(
            [
                self.block(self.block_layer_count, self.block_layer_size, self.p)
                for block in range(self.block_count)
            ]
        )

    def _make_conv_in(self):
        self.conv_in = GCNConv(
            self.in_dim, self.block_layer_size, normalize=True, cached=True
        )

    def _make_conv_out(self):
        self.conv_out = GCNConv(
            self.block_layer_size, self.out_dim, normalize=True, cached=True
        )

    def forward(self, x, edge_index):
        x = self.conv_in(x, edge_index).relu()
        x = self.drop(x)
        for block in self.blocks:
            x = block.forward(x, edge_index)
        x = self.conv_out(x, edge_index).relu()
        x = self.drop(x)
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
