import torch
from torch.nn.functional import cross_entropy
from torch_geometric.data import Data
import numpy as np

from recognition.gnn_facebookpages_s4749422_James_Brunton.visualisation import (
    get_node_embeddings,
    tsne_plot_from_embeddings,
    umap_plot_from_embeddings,
    plot_confusion_matrix_from_preds,
    plot_training_curve,
)

from recognition.gnn_facebookpages_s4749422_James_Brunton.dataset import (
    _load_edges,
    _load_targets,
    _load_features,
    make_masks,
    build_X,
    build_y,
    build_edge_index,
    build_index,
)

from recognition.gnn_facebookpages_s4749422_James_Brunton.modules import (
    GraphConvolutionalNetwork as GCN,
    GraphSAGE as GSAGE,
    blockGCN as BGCN,
)


def tempered_class_weights(y, train_mask, C, alpha=0.5):
    """
    y: torch.LongTensor [N] (any device)
    train_mask: torch.BoolTensor [N] (any device)
    """
    yy = y.detach().cpu()
    tm = train_mask.detach().cpu()

    counts = np.bincount(yy[tm].numpy(), minlength=C).astype(np.float64)
    counts = np.maximum(counts, 1)
    w = counts ** (-alpha)
    w *= C / w.sum()  # normalize mean=1
    return torch.tensor(w, dtype=torch.float32)  # caller moves to device


def main():
    edges = _load_edges()  # expects columns: id_1, id_2
    targets = _load_targets()  # expects columns: id, label
    features = _load_features()  # pd.Series: index=node_id, value=list[int]

    node_ids, nid2idx = build_index(features, targets, edges)
    idx2nid = {idx: nid for nid, idx in nid2idx.items()}  # keep for reporting
    x = build_X(features, node_ids)  # [N,F] float32
    y, label_names = build_y(targets, nid2idx)  # [N] int64 (−1 for unlabeled)
    ei = build_edge_index(edges, nid2idx)  # [2,E] int64

    train_mask, val_mask, test_mask = make_masks(y)

    data = Data(
        x=x,
        edge_index=ei,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    in_dim = data.num_features
    out_dim = int(data.y.max().item() + 1)
    model = BGCN(
        in_dim,
        out_dim,
        block_count=2,
        block_layer_count=5,
        block_layer_size=256,
        p=0.75,
        is_hour_glass=True,
        expansion_ratio=0.5,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    history = {"epoch": [], "loss": [], "val_acc": []}

    for epoch in range(1, 300):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        w = tempered_class_weights(
            y, data.train_mask, C=len(label_names), alpha=0.5
        ).to(device)
        loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(data.x, data.edge_index)
            v_pred = v_logits.argmax(dim=1)
            val_acc = (
                (v_pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            )
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["val_acc"].append(val_acc)

        if epoch % 20 == 0:
            print(f"[{epoch:03d}] loss={loss.item():.4f} val_acc={val_acc:.3f}")

    # --- final eval + predictions ---
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    test_mask = data.test_mask
    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
    print(f"\nTEST ACC: {test_acc:.3f}")

    # <<< NEW: plots — adjust output paths as you like
    # 1) Training curve
    plot_training_curve(history, out_path="figs/training_curve.png")

    # 2) Confusion matrix (test split only)
    plot_confusion_matrix_from_preds(
        y_true=data.y[test_mask].detach().cpu(),
        y_pred=pred[test_mask].detach().cpu(),
        label_names=label_names,
        normalize=True,
        title="Confusion matrix (test)",
        out_path="figs/confusion_matrix.png",
    )

    # 3) Embeddings → t-SNE / UMAP on the test split
    Z = get_node_embeddings(model, data, prefer_penultimate=True)

    tsne_plot_from_embeddings(
        Z=Z,
        y=data.y,
        mask=test_mask,
        label_names=label_names,
        title_prefix="t-SNE (test)",
        out_path="figs/tsne_test.png",
        perplexity=30,
        random_state=0,
    )

    umap_plot_from_embeddings(
        Z=Z,
        y=data.y,
        mask=test_mask,
        label_names=label_names,
        title_prefix="UMAP (test)",
        out_path="figs/umap_test.png",
        n_neighbors=15,
        min_dist=0.1,
        random_state=0,
    )


if __name__ == "__main__":
    main()
