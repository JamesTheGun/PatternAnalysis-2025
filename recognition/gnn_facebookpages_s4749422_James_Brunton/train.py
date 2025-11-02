from ast import mod
import torch
from torch.nn.functional import cross_entropy
import numpy as np

from recognition.gnn_facebookpages_s4749422_James_Brunton.visualisation import (
    generate_visuals,
)

from recognition.gnn_facebookpages_s4749422_James_Brunton.dataset import get_data

from recognition.gnn_facebookpages_s4749422_James_Brunton.modules import get_model


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
    w *= C / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def main():
    data, label_names, y = get_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    in_dim = data.num_features
    out_dim = int(data.y.max().item() + 1)
    model = get_model(in_dim=in_dim, out_dim=out_dim, device=device)

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

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    test_mask = data.test_mask
    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
    print(f"\nTEST ACC: {test_acc:.3f}")

    generate_visuals(
        data=data,
        test_mask=test_mask,
        pred=pred,
        label_names=label_names,
        model=model,
        history=history,
    )


if __name__ == "__main__":
    main()
