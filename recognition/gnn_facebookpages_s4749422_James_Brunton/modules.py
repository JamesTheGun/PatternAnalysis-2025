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
        self.conv1 = GCNConv(in_dim, hidden, normalize=True, cached=True)
        self.conv2 = GCNConv(hidden, hidden, normalize=True, cached=True)
        self.conv3 = GCNConv(hidden, out_dim, normalize=True, cached=True)
        self.drop = nn.Dropout(p)

    def forward(self, x, edge_index):
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
        expansion_ratio=1.3,
    ):
        self.block_layer_count = block_layer_count
        self.block_layer_size = block_layer_size
        self.block_count = block_count
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.is_hour_glass = is_hour_glass
        self.expansion_ratio = expansion_ratio
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
            self.layer_size = int(layer_size)
            self.block_layer_count = int(block_layer_count)
            self.p = p
            self.expansion_ratio = float(expansion_ratio)
            self.is_hour_glass = bool(is_hour_glass)
            self.drop = nn.Dropout(p)
            # Build and assign layers
            self.layers = self.build_layers()

        def build_layers(self):
            # Return a ModuleList
            if self.is_hour_glass:
                return self.build_layers_hourglass()
            else:
                return self.build_constant_layers()

        def build_constant_layers(self):
            return nn.ModuleList(
                [
                    GCNConv(
                        self.layer_size, self.layer_size, normalize=True, cached=True
                    )
                    for _ in range(self.block_layer_count)
                ]
            )

        # ----- helpers that work for ratio > 1 (grow) and ratio < 1 (shrink) -----

        def _progressive_layers(self, steps, starting_size, ratio):
            """
            Multiply channels by `ratio` for `steps`, enforcing at least +/-1 change.
            Returns (layers, final_size).
            """
            layers = []
            in_ch = int(starting_size)
            r = float(ratio)
            # Guard against degenerate ratios
            if r == 1.0:
                r = 1.0001
            for _ in range(int(steps)):
                # proposed new width
                out_ch = int(round(in_ch * r))
                # force a change of at least 1 unit toward the intended direction
                if out_ch == in_ch:
                    out_ch = in_ch + (1 if r > 1.0 else -1)
                # keep within sane bounds
                out_ch = max(1, min(out_ch, STUPIDLY_LARGE_LAYER_SIZE))
                layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                in_ch = out_ch
            return layers, in_ch

        def _towards_target_layers(self, current_size, target_size, ratio_hint):
            """
            Build layers that move monotonically from current_size to target_size.
            Uses ratio_hint (>1 means multiplicative growth per step; <1 means shrink).
            """
            layers = []
            in_ch = int(current_size)
            tgt = int(target_size)

            if in_ch == tgt:
                return layers

            # choose a multiplier > 1 for growth steps
            r = float(ratio_hint)
            if r == 1.0:
                r = 1.0001
            grow_mult = r if r > 1.0 else (1.0 / r if r > 0.0 else 2.0)

            if in_ch < tgt:
                # grow until we hit target
                while in_ch < tgt:
                    out_ch = int(round(in_ch * grow_mult))
                    if out_ch <= in_ch:
                        out_ch = in_ch + 1
                    out_ch = min(out_ch, tgt, STUPIDLY_LARGE_LAYER_SIZE)
                    layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                    in_ch = out_ch
            else:
                # shrink until we hit target
                while in_ch > tgt:
                    out_ch = int(round(in_ch / grow_mult))
                    if out_ch >= in_ch:
                        out_ch = in_ch - 1
                    out_ch = max(out_ch, tgt, 1)
                    layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                    in_ch = out_ch

            return layers

        # ----- hourglass that supports expansion_ratio < 1 -----

        def build_layers_hourglass(self):
            base = int(self.layer_size)
            up_count = self.block_layer_count // 2  # integer division

            # First half: move by multiplying with ratio (grow if >1, shrink if <1)
            up_layers, mid = self._progressive_layers(
                up_count, base, self.expansion_ratio
            )

            # Second half: return to base width monotonically
            # If ratio > 1 we will constrict; if ratio < 1 we will expand back.
            down_layers = self._towards_target_layers(
                current_size=mid, target_size=base, ratio_hint=self.expansion_ratio
            )

            return nn.ModuleList(up_layers + down_layers)

        def forward(self, x, edge_index):
            x_in = x
            for layer in self.layers:
                x = F.leaky_relu(layer(x, edge_index), negative_slope=0.1)
                x = self.drop(x)
            # residual: final width equals base `layer_size`, so shapes match
            x = x + x_in
            return x

    def _make_blocks(self):
        self.blocks = nn.ModuleList(
            [
                self.block(
                    block_layer_count=self.block_layer_count,
                    layer_size=self.block_layer_size,
                    p=self.p,
                    expansion_ratio=self.expansion_ratio,
                    is_hour_glass=self.is_hour_glass,
                )
                for _ in range(self.block_count)
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
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        return x
