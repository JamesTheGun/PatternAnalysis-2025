import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from recognition.gnn_facebookpages_s4749422_James_Brunton.parameters import VIS_OUT_PATH

# --- helpers ---------------------------------------------------------------


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _label_palette(n):
    # returns a list of RGB triples using matplotlib built-ins
    cmap = plt.cm.get_cmap("tab20", max(n, 20))
    return [cmap(i) for i in range(n)]


def _mask_or_all(mask, n):
    if mask is None:
        return np.ones(n, dtype=bool)
    return _to_numpy(mask).astype(bool)


# --- embeddings extraction -------------------------------------------------


def get_node_embeddings(model, data, prefer_penultimate=True):
    """
    Returns node embeddings for visualization.
    Tries to capture the penultimate tensor via a forward hook.
    Falls back to using logits if the penultimate cannot be captured.
    """
    model.eval()
    penultimate = {}

    if prefer_penultimate:
        last_module = None
        for m in model.modules():
            if any(
                k in m.__class__.__name__.lower()
                for k in ["linear", "conv", "sage", "gcn"]
            ):
                last_module = m
        if last_module is not None:

            def _hook(_, __, output):
                penultimate["z"] = output

            h = last_module.register_forward_hook(_hook)

            with torch.no_grad():
                _ = model(data.x, data.edge_index)

            h.remove()

    if "z" in penultimate:
        z = penultimate["z"]
        return z.detach().cpu().float()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return logits.detach().cpu().float()


def plot_2d_embedding(
    Z2d,
    y,
    mask=None,
    label_names=None,
    title="2D embedding",
    out_path=None,
    alpha=0.9,
    s=7,
):
    """
    Generic 2D scatter for node embeddings.
    Z2d: [N,2] numpy
    y:   [N] int labels (−1 allowed for unlabeled; they will be hidden)
    mask: boolean mask to select a split (e.g., data.test_mask)
    """
    Z2d = _to_numpy(Z2d)
    y = _to_numpy(y).astype(int)
    n = len(y)
    sel = _mask_or_all(mask, n) & (y >= 0)  # ignore unlabeled

    Z2d = Z2d[sel]
    y = y[sel]

    n_classes = int(y.max()) + 1 if y.size and y.max() >= 0 else 0
    palette = _label_palette(n_classes)
    label_names = label_names or [f"class {i}" for i in range(n_classes)]

    plt.figure(figsize=(7.5, 6.5))
    for c in range(n_classes):
        idx = y == c
        if not np.any(idx):
            continue
        plt.scatter(
            Z2d[idx, 0],
            Z2d[idx, 1],
            s=s,
            alpha=alpha,
            label=label_names[c],
            c=[palette[c]],
        )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if n_classes <= 20:
        plt.legend(loc="best", fontsize=8, markerscale=1.6, frameon=True)
    plt.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


def tsne_plot_from_embeddings(
    Z,
    y,
    mask=None,
    label_names=None,
    title_prefix="t-SNE",
    out_path=None,
    perplexity=30,
    random_state=0,
):
    """
    Runs t-SNE on embeddings Z [N,D] and calls plot_2d_embedding.
    """
    Z = _to_numpy(Z)
    sel = _mask_or_all(mask, len(Z))
    Zsel = Z[sel]
    ysel = _to_numpy(y)[sel]

    Z2 = TSNE(
        n_components=2,
        init="pca",
        perplexity=min(perplexity, max(5, len(Zsel) // 50)),
        random_state=random_state,
    ).fit_transform(Zsel)
    plot_2d_embedding(
        Z2,
        ysel,
        mask=None,
        label_names=label_names,
        title=f"{title_prefix} (split)",
        out_path=out_path,
    )


def umap_plot_from_embeddings(
    Z,
    y,
    mask=None,
    label_names=None,
    title_prefix="UMAP",
    out_path=None,
    n_neighbors=15,
    min_dist=0.1,
    random_state=0,
):
    """
    Runs UMAP (if installed) on embeddings Z [N,D] and calls plot_2d_embedding.
    """
    try:
        import umap
    except Exception:
        print("UMAP not installed; skipping UMAP plot.")
        return

    Z = _to_numpy(Z)
    sel = _mask_or_all(mask, len(Z))
    Zsel = Z[sel]
    ysel = _to_numpy(y)[sel]

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    Z2 = reducer.fit_transform(Zsel)
    plot_2d_embedding(
        Z2,
        ysel,
        mask=None,
        label_names=label_names,
        title=f"{title_prefix} (split)",
        out_path=out_path,
    )


# --- confusion matrix ------------------------------------------------------


def plot_confusion_matrix_from_preds(
    y_true,
    y_pred,
    label_names=None,
    normalize=True,
    title="Confusion matrix",
    out_path=None,
):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    label_names = label_names or [f"class {i}" for i in range(n_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if normalize and cm.sum(axis=1).all():
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(n_classes), label_names, rotation=45, ha="right")
    plt.yticks(range(n_classes), label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate (sparse if many classes)
    if n_classes <= 12:
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                txt = f"{val:.2f}" if normalize else str(int(val))
                plt.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if val > cm.max() * 0.6 else "white",
                )

    plt.tight_layout()
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


# --- training curve (optional) --------------------------------------------


def plot_training_curve(history, out_path=None):
    """
    history = dict with keys like "epoch", "loss", "val_acc"
    """
    if out_path:
        out_path = out_path + "//training_curve"
    ep = history.get("epoch", [])
    loss = history.get("loss", [])
    val = history.get("val_acc", [])

    plt.figure(figsize=(6.5, 4.0))
    if loss:
        plt.plot(ep, loss, label="train loss", linewidth=2)
    if val:
        plt.plot(ep, val, label="val acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training progress")
    plt.tight_layout()
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


def generate_visuals(data, test_mask, pred, label_names, model, history=None):

    if history:
        plot_training_curve(history, out_path=VIS_OUT_PATH)

    plot_confusion_matrix_from_preds(
        y_true=data.y[test_mask].detach().cpu(),
        y_pred=pred[test_mask].detach().cpu(),
        label_names=label_names,
        normalize=True,
        title="Confusion matrix",
        out_path=VIS_OUT_PATH + "//confusion_matrix",
    )

    Z = get_node_embeddings(model, data, prefer_penultimate=True)

    tsne_plot_from_embeddings(
        Z=Z,
        y=data.y,
        mask=test_mask,
        label_names=label_names,
        title_prefix="t-SNE",
        out_path=VIS_OUT_PATH + "//t_SNE",
        perplexity=30,
        random_state=0,
    )

    umap_plot_from_embeddings(
        Z=Z,
        y=data.y,
        mask=test_mask,
        label_names=label_names,
        title_prefix="UMAP",
        out_path=VIS_OUT_PATH + "//UMAP",
        n_neighbors=15,
        min_dist=0.1,
        random_state=0,
    )
