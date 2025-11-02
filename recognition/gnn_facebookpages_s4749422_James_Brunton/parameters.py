from pathlib import Path

# paths
VIS_OUT_PATH = "recognition/gnn_facebookpages_s4749422_James_Brunton/visualisations"
GCN_CHECKPOINT_PATH = str(
    Path(
        "recognition/gnn_facebookpages_s4749422_James_Brunton/checkpoints/gcn_checkpoint.pt"
    )
)

# model config
STUPIDLY_LARGE_LAYER_SIZE = (
    512  # max size we let layers get to before we raise an error
)
EPOCHS = 150
EPOCH_PRINT_INTERVAL = 20  # When do we print training status?
LEARNING_RATE = 5e-5  # model learning rate
WIEGHT_DECAY = 0.05
SELECTED_MODEL = "BGCN"  # BGCN  or "GCN" or "SAGE"

# these have already been tuned experementally
BGCN_PARAMS = {
    "block_layer_count": 5,
    "block_layer_size": 256,
    "block_count": 2,
    "p": 0.25,  # drop-out...
    "is_hour_glass": True,
    "expansion_ratio": 0.75,
}
GCN_PARAMS = {
    "hidden": 128,
    "p": 0.6,
}

GCN_SAGE_PARAMS = {
    "hidden": 128,
    "p": 0.5,
}

# data
LABELED_PER_CLASS = 128
VAL_SIZE = 300
TEST_SIZE = 5000
