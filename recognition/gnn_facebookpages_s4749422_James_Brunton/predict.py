import torch
from recognition.gnn_facebookpages_s4749422_James_Brunton.dataset import get_data
from recognition.gnn_facebookpages_s4749422_James_Brunton.modules import get_model
from recognition.gnn_facebookpages_s4749422_James_Brunton.visualisation import (
    generate_visuals,
)

from recognition.gnn_facebookpages_s4749422_James_Brunton.constants import (
    CHECKPOINT_PATH,
)


def main():
    data, label_names, y = get_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    in_dim = data.num_features
    out_dim = int(data.y.max().item() + 1)

    model = get_model(in_dim=in_dim, out_dim=out_dim, device=device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        test_mask = data.test_mask
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
        print(f"Loaded model test accuracy: {test_acc:.3f}")

    generate_visuals(
        data=data,
        test_mask=test_mask,
        pred=pred,
        label_names=label_names,
        model=model,
    )


if __name__ == "__main__":
    main()
