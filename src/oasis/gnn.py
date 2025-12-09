from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from ase import io
from ase.neighborlist import neighbor_list, natural_cutoffs
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class GraphSample:
    z: np.ndarray  # (num_nodes,)
    pos: np.ndarray  # (num_nodes, 3)
    edge_index: np.ndarray  # (num_edges, 2)
    edge_attr: np.ndarray  # (num_edges, 4) -> [dist, dx, dy, dz]
    y: float
    reaction: str


def build_edges(atoms, cutoff_mult: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edge_index, edge_attr) using PBC-aware neighbor search."""
    cutoffs = natural_cutoffs(atoms, mult=cutoff_mult)
    i, j, S = neighbor_list("ijS", atoms, cutoffs)
    cell = atoms.get_cell().array
    pos = atoms.get_positions()
    # Vector from i -> j accounting for periodic images via S
    disp = pos[j] + np.dot(S, cell) - pos[i]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    edge_index = np.stack([i, j], axis=1)
    edge_attr = np.hstack([dist, disp])

    # Make edges directed both ways so message passing is symmetric
    rev_edge_index = edge_index[:, [1, 0]]
    rev_edge_attr = np.hstack([dist, -disp])

    full_edge_index = np.concatenate([edge_index, rev_edge_index], axis=0)
    full_edge_attr = np.concatenate([edge_attr, rev_edge_attr], axis=0)
    return full_edge_index.astype(np.int64), full_edge_attr.astype(np.float32)


def load_dataset(
    xyz_path: str, parquet_path: str, cutoff_mult: float
) -> Tuple[List[GraphSample], float, float]:
    """Load atoms/labels and build graph samples."""
    frames = io.read(xyz_path, ":")
    labels_df = pd.read_parquet(parquet_path, engine="pyarrow")
    label_map: Dict[str, float] = labels_df.set_index("reaction")[
        "reference_ads_eng"
    ].to_dict()

    samples: List[GraphSample] = []
    ys: List[float] = []
    for atoms in frames:
        reaction = atoms.info.get("reaction")
        if reaction is None or reaction not in label_map:
            continue
        edge_index, edge_attr = build_edges(atoms, cutoff_mult=cutoff_mult)
        y_val = float(label_map[reaction])
        sample = GraphSample(
            z=atoms.get_atomic_numbers().astype(np.int64),
            pos=atoms.get_positions().astype(np.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_val,
            reaction=reaction,
        )
        samples.append(sample)
        ys.append(y_val)

    y_mean = float(np.mean(ys))
    y_std = float(np.std(ys) + 1e-8)
    return samples, y_mean, y_std


class AdsorptionDataset(Dataset):
    def __init__(self, samples: Sequence[GraphSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]


def collate_batch(batch: Sequence[GraphSample]):
    node_offset = 0
    zs: List[torch.Tensor] = []
    positions: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    batch_vec: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    reactions: List[str] = []

    for graph_idx, sample in enumerate(batch):
        num_nodes = sample.z.shape[0]
        zs.append(torch.from_numpy(sample.z))
        positions.append(torch.from_numpy(sample.pos))
        edge_indices.append(torch.from_numpy(sample.edge_index + node_offset))
        edge_attrs.append(torch.from_numpy(sample.edge_attr))
        batch_vec.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        y_list.append(torch.tensor([sample.y], dtype=torch.float32))
        reactions.append(sample.reaction)
        node_offset += num_nodes

    z = torch.cat(zs, dim=0)
    pos = torch.cat(positions, dim=0)
    edge_index = torch.cat(edge_indices, dim=0).t().contiguous()  # shape (2, E)
    edge_attr = torch.cat(edge_attrs, dim=0)
    batch_tensor = torch.cat(batch_vec, dim=0)
    y = torch.cat(y_list, dim=0)

    return {
        "z": z,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_tensor,
        "y": y,
        "reactions": reactions,
    }


def segment_mean(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros((num_graphs, x.size(-1)))
    counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(1)
    out.index_add_(0, batch, x)
    out = out / counts
    return out


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        # Concatenate neighbor, target, and edge features
        m_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.edge_mlp(m_in)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)

        node_in = torch.cat([x, agg], dim=-1)
        return self.node_mlp(node_in)


class AdsorptionGNN(nn.Module):
    def __init__(self, max_z: int, hidden_dim: int, layers: int, edge_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_dim) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        x = self.embedding(batch["z"])
        for layer in self.layers:
            x = layer(x, batch["edge_index"], batch["edge_attr"])
        graph_repr = segment_mean(x, batch["batch"])
        pred = self.readout(graph_repr).squeeze(-1)
        return pred


def split_indices(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    generator = rng if rng is not None else np.random.default_rng()
    idx = np.arange(n)
    generator.shuffle(idx)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    if n_train == 0 or n_train + n_val >= n:
        raise ValueError("Dataset too small for the requested split.")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def evaluate(model, loader: Optional[DataLoader], device) -> Tuple[float, float]:
    if loader is None or len(loader.dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    mae_total = 0.0
    mse_total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            preds = model(batch)
            targets = batch["y"]
            diff = preds - targets
            mae_total += diff.abs().sum().item()
            mse_total += (diff**2).sum().item()
            n += targets.numel()
    mae = mae_total / n
    rmse = math.sqrt(mse_total / n)
    return mae, rmse


def train(
    samples: List[GraphSample],
    y_mean: float,
    y_std: float,
    batch_size: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    layers: int,
    device: torch.device,
    log_interval: Optional[int] = 10,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seed(seed)

    # Normalize targets for stability without mutating caller samples
    norm_samples = [
        GraphSample(
            z=s.z,
            pos=s.pos,
            edge_index=s.edge_index,
            edge_attr=s.edge_attr,
            y=(s.y - y_mean) / y_std,
            reaction=s.reaction,
        )
        for s in samples
    ]

    max_z = int(max(s.z.max() for s in samples))
    split_rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = split_indices(
        len(samples), train_frac=train_frac, val_frac=val_frac, rng=split_rng
    )

    train_ds = AdsorptionDataset([norm_samples[i] for i in train_idx])
    val_ds = AdsorptionDataset([norm_samples[i] for i in val_idx])
    test_ds = AdsorptionDataset([norm_samples[i] for i in test_idx])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )
        if len(val_ds) > 0
        else None
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = AdsorptionGNN(max_z=max_z, hidden_dim=hidden_dim, layers=layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            preds = model(batch)
            loss = loss_fn(preds, batch["y"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["y"].numel()

        train_mae = epoch_loss / len(train_ds)
        val_mae, val_rmse = evaluate(model, val_loader, device)
        if val_loader is not None and val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if log_interval is not None and (
            epoch % log_interval == 0 or epoch == 1 or epoch == epochs
        ):
            print(
                f"Epoch {epoch:03d} | train MAE (normalized): {train_mae:.4f} | "
                f"val MAE: {val_mae:.4f} | val RMSE: {val_rmse:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set and denormalize
    test_mae, test_rmse = evaluate(model, test_loader, device)
    test_mae *= y_std
    test_rmse *= y_std
    val_mae, val_rmse = evaluate(model, val_loader, device)
    val_mae *= y_std
    val_rmse *= y_std

    return (
        model,
        {
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        },
        y_mean,
        y_std,
    )


def evaluate_train_test_splits(
    samples: Sequence[GraphSample],
    y_mean: float,
    y_std: float,
    train_fracs: Sequence[float],
    device: torch.device,
    *,
    batch_size: int = 16,
    epochs: int = 150,
    lr: float = 5e-4,
    hidden_dim: int = 128,
    layers: int = 3,
    seed: int = 0,
    log_interval: Optional[int] = None,
) -> pd.DataFrame:
    """Train and evaluate the GNN across multiple train/test splits."""
    results = []
    total = len(samples)
    for run_idx, train_frac in enumerate(train_fracs):
        if not 0 < train_frac < 1:
            raise ValueError(
                f"Invalid train fraction {train_frac}; must be between 0 and 1."
            )
        split_seed = seed + run_idx
        _, metrics, _, _ = train(
            samples=list(samples),
            y_mean=y_mean,
            y_std=y_std,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            layers=layers,
            device=device,
            log_interval=log_interval,
            train_frac=train_frac,
            val_frac=0.0,
            seed=split_seed,
        )
        n_train = int(train_frac * total)
        n_test = total - n_train
        results.append(
            {
                "train_frac": train_frac,
                "test_frac": 1 - train_frac,
                "n_train": n_train,
                "n_test": n_test,
                "val_mae": float(metrics["val_mae"]),
                "val_rmse": float(metrics["val_rmse"]),
                "test_mae": float(metrics["test_mae"]),
                "test_rmse": float(metrics["test_rmse"]),
            }
        )
    return pd.DataFrame(results)


def evaluate_splits_from_files(
    xyz_path: str,
    parquet_path: str,
    train_fracs: Sequence[float],
    device: torch.device,
    *,
    cutoff_mult: float = 1.2,
    batch_size: int = 16,
    epochs: int = 6,
    lr: float = 5e-4,
    hidden_dim: int = 64,
    layers: int = 2,
    seed: int = 0,
    log_interval: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load data from disk and run train/test split evaluations using default hyperparameters.
    """
    samples, y_mean, y_std = load_dataset(
        xyz_path, parquet_path, cutoff_mult=cutoff_mult
    )
    return evaluate_train_test_splits(
        samples=samples,
        y_mean=y_mean,
        y_std=y_std,
        train_fracs=train_fracs,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        layers=layers,
        seed=seed,
        log_interval=log_interval,
    )
