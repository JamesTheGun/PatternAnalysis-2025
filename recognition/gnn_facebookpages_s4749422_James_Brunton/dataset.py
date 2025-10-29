import pandas as pd
import os


def _get_data_directory():
    return os.path.join(
        "recognition",
        "gnn_facebookpages_s4749422_James_Brunton",
        "data",
        "facebook_large",
    )


def _load_edges():
    return pd.read_csv(os.path.join(_get_data_directory(), "musae_facebook_edges.csv"))


def _load_targets():
    return pd.read_csv(os.path.join(_get_data_directory(), "musae_facebook_target.csv"))


def _load_features():
    return pd.read_json(
        os.path.join(_get_data_directory(), "musae_facebook_features.json"),
        typ="series",
    )