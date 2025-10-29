import pandas as pd
import torch
import numpy as np
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


def build_X(features, node_ids):
    # features: Series[node_id -> list[int]]
    s = features.reindex(node_ids).apply(lambda x: x if isinstance(x, list) else [])
    s = s.apply(lambda toks: list(set(toks)))  # dedupe
    max_tok = max((max(t) if t else 0) for t in s) if len(s) else 0
    F = max_tok + 1
    X = np.zeros((len(node_ids), F), dtype=np.float32)  # simplest (dense)
    for i, toks in enumerate(s):
        if toks:
            X[i, toks] = 1.0
    return torch.from_numpy(X)


def build_y(targets, nid2idx):
    labels = sorted(targets["label"].unique())
    lab2i = {lab: i for i, lab in enumerate(labels)}
    N = len(nid2idx)
    y = -np.ones(N, dtype=np.int64)
    for nid, lab in zip(targets["id"], targets["label"]):
        if nid in nid2idx:
            y[nid2idx[nid]] = lab2i[lab]
    return torch.from_numpy(y), labels


def build_edge_index(edges, nid2idx):
    src = edges["id_1"].map(nid2idx).to_numpy()
    dst = edges["id_2"].map(nid2idx).to_numpy()
    e0 = np.concatenate([src, dst])
    e1 = np.concatenate([dst, src])
    mask = e0 != e1
    edge_index = np.vstack([e0[mask], e1[mask]]).astype(np.int64)  # [2,E]
    return torch.from_numpy(edge_index)


def make_masks(y, per_class_train=20, val_size=500, test_size=1000, seed=0):
    N = y.shape[0]
    rng = np.random.default_rng(seed)
    train = np.zeros(N, dtype=bool)
    val = np.zeros(N, dtype=bool)
    test = np.zeros(N, dtype=bool)

    labeled = np.where(y.numpy() >= 0)[0]
    classes = np.unique(y.numpy()[labeled])

    for c in classes:
        idx_c = labeled[y.numpy()[labeled] == c]
        rng.shuffle(idx_c)
        take = min(per_class_train, len(idx_c))
        train[idx_c[:take]] = True

    rest = labeled[~train[labeled]]
    rng.shuffle(rest)
    val[rest[:val_size]] = True
    test[rest[val_size : val_size + test_size]] = True
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)
