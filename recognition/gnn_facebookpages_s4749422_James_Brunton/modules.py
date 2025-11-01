from logging import raiseExceptions
from re import L
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, Sequential
import torch.nn.functional as F

STUPIDLY_LARGE_LAYER_SIZE = 512


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
        is_hour_glass=False,
    ):
        self.block_layer_count = block_layer_count
        self.block_layer_size = block_layer_size
        self.block_count = block_count
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.is_hour_glass = is_hour_glass
        super().__init__()
        self._construct_shit()
        self.drop = nn.Dropout(p)

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
            expansion_ratio=1.4,
            is_hour_glass=False,
        ):
            super().__init__()
            self.layer_size = layer_size
            self.block_layer_count = block_layer_count
            self.last_layer_size = layer_size
            self.p = p
            self.expansion_ratio = expansion_ratio
            self.is_hour_glass = is_hour_glass

            print(self.is_hour_glass)
            self.layers = self.build_layers()
            self.drop = nn.Dropout(p)

        def build_layers(self):
            print("bruh 2/./0")
            if self.is_hour_glass:
                print("bruh")
                self.build_layers_hourglass()
            else:
                self.build_constant_layers()

        def build_constant_layers(self):
            self.layers = nn.ModuleList(
                [
                    GCNConv(
                        self.layer_size, self.layer_size, normalize=True, cached=True
                    )
                    for layer in range(self.block_layer_count)
                ]
            )

        def build_expansion_layers(self, expansion_layer_count, starting_size):
            expansion_layers = []
            this_layer_size = starting_size
            for layer in range(expansion_layer_count):
                next_layer_size = this_layer_size * self.expansion_ratio
                this_layer = GCNConv(
                    this_layer_size, next_layer_size, normalize=True, cached=True
                )
                this_layer_size = next_layer_size
                if next_layer_size > STUPIDLY_LARGE_LAYER_SIZE:
                    raise ValueError(
                        "You passed a silly expansion ratio for the number of layers you want... make it smaller!"
                    )
                expansion_layers.append(this_layer)
            return expansion_layers, this_layer_size

        def build_constriction_layers(self, current_size, ending_size):
            layers = []
            in_ch = int(round(current_size))
            end_ch = int(round(ending_size))
            ratio = float(self.expansion_ratio)
            if ratio <= 1.0:
                ratio = 1.0001

            while in_ch > end_ch:
                out_ch = max(end_ch, int(round(in_ch / ratio)))
                if out_ch == in_ch:
                    out_ch = max(end_ch, in_ch - 1)
                layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                in_ch = out_ch

            return layers

        def build_layers_hourglass(self):
            this_layer_size = self.layer_size
            expansion_layer_count = self.block_layer_count / 2
            expansion_layers, expansion_ending_size = self.build_expansion_layers(
                expansion_layer_count, this_layer_size
            )
            constriction_layers = self.build_constriction_layers(
                expansion_ending_size, self.layer_size
            )
            all_layers = expansion_layers + constriction_layers
            print("wtf")
            self.layers = nn.ModuleList(all_layers)

        def forward(self, x, edge_index):
            x_in = x
            for layer in self.layers:
                x = F.leaky_relu(layer(x, edge_index), negative_slope=0.1)
                x = self.drop(x)
            x = x + x_in
            return x

    def _make_blocks(self):
        self.blocks = nn.ModuleList(
            [
                self.block(
                    self.block_layer_count,
                    self.block_layer_size,
                    self.p,
                    self.is_hour_glass,
                )
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
