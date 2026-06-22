from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oasis.config import MoETrainingConfig
from oasis.learning_curve.execution import require_min_mlip_feature_count
from oasis.learning_curve.families.gating_policy import DenseGatingPolicy, GatingPolicy
from oasis.learning_curve.families.gnn_gate import collate_graphs_with_distances
from oasis.learning_curve.families.torch_device import (
    resolve_torch_device,
    state_dict_to_cpu,
    tensor_to_numpy,
    tensors_to_device,
)
from oasis.sweep import GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import SelectionRefitPolicy, resolved_training_epochs


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device),
    )
    return out / count.clamp(min=1)


class SchNetInteraction(nn.Module):
    """Single SchNet interaction block.

    Computes a continuous-filter convolution: per-edge distance filters
    (produced from RBF features via a two-layer MLP with SiLU) modulate
    neighbour embeddings, which are then scatter-summed to each atom and
    mixed back with a residual linear.
    """

    def __init__(self, hidden_dim: int, n_rbf: int) -> None:
        super().__init__()
        # Maps RBF features → per-edge filter vector (the cfconv filter network).
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: Tensor,           # (n_nodes, hidden_dim)
        edge_index: Tensor,  # (2, n_edges) long
        rbf: Tensor,         # (n_edges, n_rbf)
    ) -> Tensor:             # (n_nodes, hidden_dim)
        edge_src, edge_dst = edge_index[0], edge_index[1]
        n_nodes, hidden_dim = h.shape

        W = self.filter_net(rbf)              # (n_edges, hidden_dim)
        messages = h[edge_src] * W            # (n_edges, hidden_dim)

        agg = torch.zeros(n_nodes, hidden_dim, dtype=h.dtype, device=h.device)
        agg.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(messages), messages)

        return h + self.output_linear(agg)    # residual connection


class SchNetEncoder(nn.Module):
    """SchNet-style encoder: atom-type embeddings + continuous-filter interactions.

    Accepts integer atomic numbers rather than raw float node features,
    giving each element a learned dense embedding. Per-edge Gaussian RBF
    expansion of interatomic distances drives the filter network in each
    interaction block, so gating decisions are physically informed by
    actual bond lengths.

    Args:
        hidden_dim:     Width of atom embeddings and interaction layers.
        out_features:   Number of output logits (= n_experts for MoE gating).
        n_layers:       Number of SchNet interaction blocks.
        n_rbf:          Number of Gaussian RBF basis functions.
        r_max:          Distance cutoff (Å); RBF centers span [0, r_max].
        max_atomic_num: Largest atomic number accepted (inclusive).
    """

    def __init__(
        self,
        hidden_dim: int,
        out_features: int,
        n_layers: int,
        n_rbf: int = 20,
        r_max: float = 6.0,
        max_atomic_num: int = 100,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_num + 1, hidden_dim)

        centers = torch.linspace(0.0, r_max, n_rbf)
        width = torch.tensor(r_max / (n_rbf - 1) if n_rbf > 1 else float(r_max))
        self.register_buffer("_rbf_centers", centers)
        self.register_buffer("_rbf_width", width)

        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden_dim, n_rbf) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def _rbf_expand(self, distances: Tensor) -> Tensor:
        """Expand (n_edges,) distances → (n_edges, n_rbf) Gaussian features."""
        diff = distances.unsqueeze(1) - self._rbf_centers  # (n_edges, n_rbf)
        return torch.exp(-0.5 * (diff / self._rbf_width) ** 2)

    def forward(
        self,
        atomic_numbers: Tensor,   # (total_nodes,) long
        edge_index: Tensor,        # (2, total_edges) long
        edge_distances: Tensor,    # (total_edges,) float32
        batch_vector: Tensor,      # (total_nodes,) long
    ) -> Tensor:                   # (n_graphs, out_features)
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.embedding(atomic_numbers)       # (total_nodes, hidden_dim)
        rbf = self._rbf_expand(edge_distances)   # (total_edges, n_rbf)

        for interaction in self.interactions:
            h = interaction(h, edge_index, rbf)

        pooled = _global_mean_pool(h, batch_vector, n_graphs)  # (n_graphs, hidden_dim)
        return self.output_proj(pooled)                         # (n_graphs, out_features)


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------

def _collate_schnet(
    graphs: Sequence[GraphRecord],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate GraphRecords into SchNetEncoder inputs.

    node_features[:, 0] holds atomic numbers (float from atoms_to_graph_record);
    converts them to long for the embedding table.

    Returns:
        atomic_numbers: (total_nodes,) long
        edge_index:     (2, total_edges) long, with per-graph node offsets
        edge_distances: (total_edges,) float32
        batch_vector:   (total_nodes,) long
    """
    node_features, edge_index, batch_vector, edge_distances = collate_graphs_with_distances(graphs)
    atomic_numbers = node_features[:, 0].long()
    return atomic_numbers, edge_index, edge_distances, batch_vector


def _graphs_from_dataset(dataset: SweepDataset) -> list[GraphRecord]:
    return [dataset.graphs[sid] for sid in dataset.sample_ids.tolist()]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_schnet_encoder(
    encoder: SchNetEncoder,
    atomic_numbers: Tensor,
    edge_index: Tensor,
    edge_distances: Tensor,
    batch_vector: Tensor,
    X: Tensor,
    y: Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    encoder.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = encoder(atomic_numbers, edge_index, edge_distances, batch_vector)
        weights = torch.softmax(logits, dim=1)
        preds = (X * weights).sum(dim=1)
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Gate model and tuning spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SchNetGateModel:
    """Frozen snapshot of a trained SchNetEncoder used for MoE gating."""

    state_dict: dict[str, Any]
    n_experts: int
    hidden_dim: int
    n_layers: int
    n_rbf: int = 20
    r_max: float = 6.0
    max_atomic_num: int = 100
    bias: float = 0.0
    device: str = "cpu"
    policy: GatingPolicy = field(default_factory=DenseGatingPolicy)

    def _build_encoder(self) -> SchNetEncoder:
        device = resolve_torch_device(self.device)
        encoder = SchNetEncoder(
            hidden_dim=self.hidden_dim,
            out_features=self.n_experts,
            n_layers=self.n_layers,
            n_rbf=self.n_rbf,
            r_max=self.r_max,
            max_atomic_num=self.max_atomic_num,
        )
        encoder.load_state_dict(self.state_dict)
        encoder.eval()
        return encoder.to(device)

    def predict(self, X: np.ndarray, graphs: Sequence[GraphRecord]) -> np.ndarray:
        encoder = self._build_encoder()
        atomic_numbers, edge_index, edge_distances, batch_vector = _collate_schnet(graphs)
        device = resolve_torch_device(self.device)
        atomic_numbers, edge_index, edge_distances, batch_vector = tensors_to_device(
            device,
            atomic_numbers,
            edge_index,
            edge_distances,
            batch_vector,
        )
        with torch.no_grad():
            logits = encoder(atomic_numbers, edge_index, edge_distances, batch_vector)
        weights = self.policy.apply(tensor_to_numpy(logits))
        return (X * weights).sum(axis=1) + self.bias


@dataclass(frozen=True, slots=True)
class SchNetGateTuningSpec:
    """Tune a SchNetEncoder-based MoE gate via Optuna trial objectives."""

    training_cfg: MoETrainingConfig
    hidden_dims: tuple[int, ...] = ()   # non-empty = fixed arch; empty = search
    n_rbf: int = 20
    r_max: float = 6.0
    policy: GatingPolicy = field(default_factory=DenseGatingPolicy)

    def _arch_from_trial(self, trial: Any) -> tuple[int, int]:
        """Return (hidden_dim, n_layers) from fixed config or Optuna suggestions."""
        if self.hidden_dims:
            return self.hidden_dims[0], len(self.hidden_dims)
        hidden_dim: int = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        n_layers: int = trial.suggest_int("n_layers", 1, 3)
        return hidden_dim, n_layers

    def build_trial_objective(
        self,
        split: TrainValTestSweepRunnerInput,
    ) -> Callable[[Any], float]:
        subsets = split.dataset_subsets()
        train_ds = subsets.train
        val_ds = subsets.val

        train_graphs = _graphs_from_dataset(train_ds)
        val_graphs = _graphs_from_dataset(val_ds)
        n_experts = require_min_mlip_feature_count(
            split.dataset.mlip_features,
            min_features=2,
            method_name="moe",
        )

        train_z, train_ei, train_ed, train_bv = _collate_schnet(train_graphs)
        val_z, val_ei, val_ed, val_bv = _collate_schnet(val_graphs)
        X_train = torch.tensor(train_ds.mlip_features, dtype=torch.float32)
        y_train = torch.tensor(train_ds.targets, dtype=torch.float32)
        X_val_np = val_ds.mlip_features
        y_val_np = val_ds.targets

        seed = self.training_cfg.seed
        device = resolve_torch_device(self.training_cfg.device)
        train_z, train_ei, train_ed, train_bv, X_train, y_train = tensors_to_device(
            device,
            train_z,
            train_ei,
            train_ed,
            train_bv,
            X_train,
            y_train,
        )
        n_rbf = self.n_rbf
        r_max = self.r_max
        policy = self.policy

        def objective(trial: Any) -> float:
            hidden_dim, n_layers = self._arch_from_trial(trial)
            epochs = resolved_training_epochs(self.training_cfg, trial)
            lr: float = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            if seed is not None:
                torch.manual_seed(seed)
            encoder = SchNetEncoder(
                hidden_dim,
                n_experts,
                n_layers,
                n_rbf=n_rbf,
                r_max=r_max,
            ).to(device)
            _train_schnet_encoder(
                encoder, train_z, train_ei, train_ed, train_bv, X_train, y_train,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
            )
            val_z_device, val_ei_device, val_ed_device, val_bv_device = tensors_to_device(
                device,
                val_z,
                val_ei,
                val_ed,
                val_bv,
            )
            encoder.eval()
            with torch.no_grad():
                logits = encoder(val_z_device, val_ei_device, val_ed_device, val_bv_device)
            weights = policy.apply(tensor_to_numpy(logits))
            preds = (X_val_np * weights).sum(axis=1)
            return float(np.sqrt(np.mean((y_val_np - preds) ** 2)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> SchNetGateModel:
        subsets = split.dataset_subsets()

        if refit_policy == "train_only":
            fit_graphs = _graphs_from_dataset(subsets.train)
            X_np = subsets.train.mlip_features
            y_np = subsets.train.targets
        else:
            fit_graphs = (
                _graphs_from_dataset(subsets.train) + _graphs_from_dataset(subsets.val)
            )
            X_np = np.concatenate([subsets.train.mlip_features, subsets.val.mlip_features])
            y_np = np.concatenate([subsets.train.targets, subsets.val.targets])

        n_experts = require_min_mlip_feature_count(
            split.dataset.mlip_features,
            min_features=2,
            method_name="moe",
        )
        hidden_dim, n_layers = self._arch_from_trial(best_trial)
        epochs = resolved_training_epochs(self.training_cfg, best_trial)
        lr: float = best_trial.params["lr"]
        weight_decay: float = best_trial.params["weight_decay"]

        z, ei, ed, bv = _collate_schnet(fit_graphs)
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        device = resolve_torch_device(self.training_cfg.device)
        z, ei, ed, bv, X, y = tensors_to_device(device, z, ei, ed, bv, X, y)

        if self.training_cfg.seed is not None:
            torch.manual_seed(self.training_cfg.seed)
        encoder = SchNetEncoder(
            hidden_dim,
            n_experts,
            n_layers,
            n_rbf=self.n_rbf,
            r_max=self.r_max,
        ).to(device)
        _train_schnet_encoder(
            encoder, z, ei, ed, bv, X, y,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
        )

        encoder.eval()
        with torch.no_grad():
            logits = encoder(z, ei, ed, bv)
        weights = self.policy.apply(tensor_to_numpy(logits))
        preds = (X_np * weights).sum(axis=1)
        bias = float(np.mean(y_np - preds))

        return SchNetGateModel(
            state_dict=state_dict_to_cpu(encoder.state_dict()),
            n_experts=n_experts,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_rbf=self.n_rbf,
            r_max=self.r_max,
            bias=bias,
            device=str(device),
            policy=self.policy,
        )

    def predict(self, model: SchNetGateModel, dataset: SweepDataset) -> np.ndarray:
        return model.predict(dataset.mlip_features, _graphs_from_dataset(dataset))

    def trial_metadata(self, best_trial: Any, model: SchNetGateModel) -> dict[str, Any]:
        return {
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "n_experts": model.n_experts,
            "epochs": resolved_training_epochs(self.training_cfg, best_trial),
            "n_rbf": model.n_rbf,
            "r_max": model.r_max,
            "bias": model.bias,
        }
