import torch
from torch.nn.functional import cross_entropy, softmax

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
from torch_geometric.data import Data


# --- main: as small as possible ---
def main():
    edges = _load_edges()  # expects columns: id_1, id_2
    targets = _load_targets()  # expects columns: id, label
    features = _load_features()  # pd.Series: index=node_id, value=list[int]

    node_ids, nid2idx = build_index(features, targets, edges)
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
        in_dim, out_dim, block_count=2, block_layer_count=3, block_layer_size=64
    )
    model._make_blocks()
    # model = GCN(in_dim, hidden=128, out_dim=out_dim, p=0.75).to(device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

    for epoch in range(1, 200):
        # print("epoch do be epoching")
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                val_acc = (
                    (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                )
            print(f"[{epoch:03d}] loss={loss.item():.4f} val_acc={val_acc:.3f}")

    # final test
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        test_acc = (
            (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        )
    print(f"TEST ACC: {test_acc:.3f}")

    torch.save(
        {"state_dict": model.state_dict(), "labels": label_names}, "gcn_checkpoint.pt"
    )
    print("saved gcn_checkpoint.pt")

    sample_nids = [554, 9218, 10772, 9283, 3324, 1111, 123, 9217]
    with torch.no_grad():
        probs = softmax(logits, dim=1)
        for nid in sample_nids:
            if nid not in nid2idx:
                continue
            i = nid2idx[nid]
            top = torch.topk(probs[i], k=min(3, probs.shape[1]))
            print(f"\nNode {nid}:")
            for p, c in zip(top.values.tolist(), top.indices.tolist()):
                print(f"  {label_names[c]}: {p:.3f}")


if __name__ == "__main__":
    main()
