import pandas as pd
import torch
import numpy as np
import os
from torch_geometric.data import Data

from recognition.gnn_facebookpages_s4749422_James_Brunton.parameters import (
    LABELED_PER_CLASS,
    VAL_SIZE,
    TEST_SIZE,
)


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


def _build_X(features, node_ids):
    s = features.reindex(node_ids).apply(lambda x: x if isinstance(x, list) else [])
    s = s.apply(lambda toks: list(set(toks)))
    max_tok = max((max(t) if t else 0) for t in s) if len(s) else 0
    F = max_tok + 1
    X = np.zeros((len(node_ids), F), dtype=np.float32)
    for i, toks in enumerate(s):
        if toks:
            X[i, toks] = 1.0
    return torch.from_numpy(X)


def _build_y(targets, nid2idx):
    labels = sorted(targets["page_type"].unique())
    lab2i = {lab: i for i, lab in enumerate(labels)}
    N = len(nid2idx)
    y = -np.ones(N, dtype=np.int64)
    for nid, lab in zip(targets["id"], targets["page_type"]):
        if nid in nid2idx:
            y[nid2idx[nid]] = lab2i[lab]
    return torch.from_numpy(y), labels


def _build_edge_index(edges, nid2idx):
    src = edges["id_1"].map(nid2idx).to_numpy()
    dst = edges["id_2"].map(nid2idx).to_numpy()
    e0 = np.concatenate([src, dst])
    e1 = np.concatenate([dst, src])
    mask = e0 != e1
    edge_index = np.vstack([e0[mask], e1[mask]]).astype(np.int64)
    return torch.from_numpy(edge_index)


def _build_index(features, targets, edges):
    node_ids = np.array(
        sorted(
            set(features.index)
            | set(targets["id"])
            | set(edges["id_1"])
            | set(edges["id_2"])
        )
    )
    nid2idx = {nid: i for i, nid in enumerate(node_ids)}
    return node_ids, nid2idx


def _make_masks(
    y, per_class_train=LABELED_PER_CLASS, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=0
):
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


def get_data():
    edges = _load_edges()
    targets = _load_targets()
    features = _load_features()

    node_ids, nid2idx = _build_index(features, targets, edges)
    idx2nid = {idx: nid for nid, idx in nid2idx.items()}
    x = _build_X(features, node_ids)
    y, label_names = _build_y(targets, nid2idx)
    ei = _build_edge_index(edges, nid2idx)

    train_mask, val_mask, test_mask = _make_masks(y)

    data = Data(
        x=x,
        edge_index=ei,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return (
        data,
        label_names,
        y,
    )
