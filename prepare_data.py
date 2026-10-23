"""
Regenerate cora_data.npz — the precomputed Cora graph used at inference time.

Run this once, locally, whenever the dataset changes. It is NOT needed to serve
the API: it exists so the deployed service does not have to carry PyTorch and
PyTorch Geometric (~800MB) just to load a 2,708-node graph.

    pip install torch torch-geometric
    python prepare_data.py
"""
import numpy as np
from torch_geometric.datasets import Planetoid

data = Planetoid(root="data/Planetoid", name="Cora")[0]

np.savez_compressed(
    "cora_data.npz",
    x=data.x.numpy().astype("float32"),
    edge_index=data.edge_index.numpy().astype("int64"),
    y=data.y.numpy().astype("int64"),
    train_mask=data.train_mask.numpy(),
    val_mask=data.val_mask.numpy(),
    test_mask=data.test_mask.numpy(),
)

print(f"saved cora_data.npz — {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
