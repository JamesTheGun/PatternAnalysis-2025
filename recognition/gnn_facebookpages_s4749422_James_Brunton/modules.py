import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F

from recognition.gnn_facebookpages_s4749422_James_Brunton.parameters import (
    STUPIDLY_LARGE_LAYER_SIZE,
    BGCN_PARAMS,
    GCN_PARAMS,
    GCN_SAGE_PARAMS,
)


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

    # Nested class representing a block in the model
    # supports expansion ratios greater than zero
    # if less than 1, we have a unet, otherwise we have
    # hour glass archetecture. Also supports no expansion
    # ratio/hourglass == None if we want constant layer
    # size...
    class block(nn.Module):
        def __init__(
            self,
            block_layer_count=5,
            layer_size=64,
            p=0.75,
            expansion_ratio=1.4,
            is_hour_glass=True,
        ):
            super().__init__()
            self.layer_size = int(layer_size)
            self.block_layer_count = int(block_layer_count)
            self.p = p
            self.expansion_ratio = float(expansion_ratio)
            self.is_hour_glass = bool(is_hour_glass)
            self.drop = nn.Dropout(p)
            self.layers = self._build_layers()

        def _build_layers(self):
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

        def _progressive_layers(self, steps, starting_size, ratio):
            layers = []
            in_ch = int(starting_size)
            r = float(ratio)
            if r == 1.0:
                r = 1.0001
            for _ in range(int(steps)):
                out_ch = int(round(in_ch * r))
                if out_ch == in_ch:
                    out_ch = in_ch + (1 if r > 1.0 else -1)
                out_ch = max(1, min(out_ch, STUPIDLY_LARGE_LAYER_SIZE))
                layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                in_ch = out_ch
            return layers, in_ch

        def _towards_target_layers(self, current_size, target_size, ratio_hint):
            # shinks/expands back towards a target layer size...
            layers = []
            in_ch = int(current_size)
            tgt = int(target_size)

            if in_ch == tgt:
                return layers

            r = float(ratio_hint)
            if r == 1.0:
                r = 1.0001
            grow_mult = r if r > 1.0 else (1.0 / r if r > 0.0 else 2.0)

            if in_ch < tgt:
                while in_ch < tgt:
                    out_ch = int(round(in_ch * grow_mult))
                    if out_ch <= in_ch:
                        out_ch = in_ch + 1
                    out_ch = min(out_ch, tgt, STUPIDLY_LARGE_LAYER_SIZE)
                    layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                    in_ch = out_ch
            else:
                while in_ch > tgt:
                    out_ch = int(round(in_ch / grow_mult))
                    if out_ch >= in_ch:
                        out_ch = in_ch - 1
                    out_ch = max(out_ch, tgt, 1)
                    layers.append(GCNConv(in_ch, out_ch, normalize=True, cached=True))
                    in_ch = out_ch

            return layers

        def build_layers_hourglass(self):
            base = int(self.layer_size)
            up_count = self.block_layer_count // 2

            up_layers, mid = self._progressive_layers(
                up_count, base, self.expansion_ratio
            )

            down_layers = self._towards_target_layers(
                current_size=mid, target_size=base, ratio_hint=self.expansion_ratio
            )

            return nn.ModuleList(up_layers + down_layers)

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
                    block_layer_count=self.block_layer_count,
                    layer_size=self.block_layer_size,
                    p=self.p,
                    expansion_ratio=self.expansion_ratio,
                    is_hour_glass=self.is_hour_glass,
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


def get_model(in_dim, out_dim, device, model_type="bgcn", **kwargs):
    model_type = str(model_type).lower()

    if model_type in {"bgcn", "blockgcn"}:
        params = BGCN_PARAMS.copy()
        params.update(kwargs)
        model = blockGCN(in_dim=in_dim, out_dim=out_dim, **params)

    elif model_type == "gcn":
        params = GCN_PARAMS.copy()
        params.update(kwargs)
        model = GraphConvolutionalNetwork(
            in_dim=in_dim,
            out_dim=out_dim,
            **params,
        )

    elif model_type in {"sage", "graphsage"}:
        params = GCN_SAGE_PARAMS.copy()
        params.update(kwargs)
        model = GraphSAGE(
            in_dim=in_dim,
            out_dim=out_dim,
            **params,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)
