from pathlib import Path

# paths
VIS_OUT_PATH = "recognition/gnn_facebookpages_s4749422_James_Brunton/visualisations"
GCN_CHECKPOINT_PATH = str(
    Path(
        "recognition/gnn_facebookpages_s4749422_James_Brunton/checkpoints/gcn_checkpoint.pt"
    )
)

# model
STUPIDLY_LARGE_LAYER_SIZE = 512
EPOCHS = 5
EPOCH_PRINT_INTERVAL = 20
LEARNING_RATE = 1e-4
WIEGHT_DECAY = 0.1
BGCN_PARAMS = {
    "block_layer_count": 5,
    "block_layer_size": 256,
    "block_count": 2,
    "p": 0.75,
    "is_hour_glass": True,
    "expansion_ratio": 0.5,
}
GCN_PARAMS = {
    "hidden": 128,
    "p": 0.6,
}

SAGE_PARAMS = {
    "hidden": 128,
    "p": 0.5,
}
# data
LABELED_PER_CLASS = 128
VAL_SIZE = 500
TEST_SIZE = 1000
