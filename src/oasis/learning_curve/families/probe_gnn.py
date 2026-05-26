from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oasis.config import MoETrainingConfig
from oasis.learning_curve.families.gnn_gate import collate_graphs
from oasis.sweep import GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import SelectionRefitPolicy


def _scatter_mean(src: Tensor, idx: Tensor, n_bins: int) -> Tensor:
    """Mean-aggregate src rows into n_bins buckets indexed by idx."""
    hidden = src.shape[1]
    out = torch.zeros(n_bins, hidden, dtype=src.dtype, device=src.device)
    if src.shape[0] == 0:
        return out
    out.scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)
    count = torch.zeros(n_bins, 1, dtype=src.dtype, device=src.device)
    count.scatter_add_(
        0,
        idx.unsqueeze(1),
        torch.ones(idx.shape[0], 1, dtype=src.dtype, device=src.device),
    )
    return out / count.clamp(min=1)


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device),
    )
    return out / count.clamp(min=1)


class ProbeGnnEncoder(nn.Module):
    """Message-passing GNN for direct adsorption energy regression.

    Accepts probe-augmented float node features (column 0: atomic number;
    columns 1..: per-atom MLIP probe energies, zero for non-binding atoms),
    runs ``n_layers`` of scatter-mean message-passing steps, global-mean-pools
    the resulting node embeddings, and projects to a scalar energy per graph.

    Args:
        in_features: Width of each node feature vector (1 + n_mlips after
            probe augmentation).
        hidden_dim:  Width of all hidden layers.
        n_layers:    Number of message-passing iterations.

    Forward signature::

        forward(node_features, edge_index, batch_vector) -> Tensor

    Returns:
        ``(n_graphs, 1)`` float32 tensor of predicted adsorption energies.
    """

    def __init__(self, in_features: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.mp_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: Tensor,  # (total_nodes, in_features) float32
        edge_index: Tensor,     # (2, total_edges) long
        batch_vector: Tensor,   # (total_nodes,) long
    ) -> Tensor:                # (n_graphs, 1) float32
        n_nodes = node_features.shape[0]
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.input_proj(node_features)

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        for layer in self.mp_linears:
            agg = _scatter_mean(h[edge_src], edge_dst, n_nodes)
            h = F.relu(layer(h + agg))

        pooled = _global_mean_pool(h, batch_vector, n_graphs)
        return self.output_proj(pooled)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_probe_encoder(
    encoder: ProbeGnnEncoder,
    node_features: Tensor,
    edge_index: Tensor,
    batch_vector: Tensor,
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
        preds = encoder(node_features, edge_index, batch_vector).squeeze(1)
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.step()


def _graphs_from_dataset(dataset: SweepDataset) -> list[GraphRecord]:
    probe_records = (dataset.auxiliary_views or {}).get("probe_gnn_records")
    if probe_records is None:
        raise ValueError(
            "probe_gnn requires probe-augmented graph records in "
            "dataset.auxiliary_views['probe_gnn_records']; configure "
            "probe_features.dataset_path so the method can load them."
        )
    return list(probe_records)


def _direct_graphs_from_dataset(dataset: SweepDataset) -> list[GraphRecord]:
    return [dataset.graphs[sid] for sid in dataset.sample_ids.tolist()]


# ---------------------------------------------------------------------------
# Frozen model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProbeGnnModel:
    """Frozen snapshot of a trained ProbeGnnEncoder for direct energy prediction."""

    state_dict: dict[str, Any]
    in_features: int
    hidden_dim: int
    n_layers: int
    bias: float = 0.0

    def _build_encoder(self) -> ProbeGnnEncoder:
        encoder = ProbeGnnEncoder(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )
        encoder.load_state_dict(self.state_dict)
        encoder.eval()
        return encoder

    def predict(self, graphs: Sequence[GraphRecord]) -> np.ndarray:
        """Return predicted adsorption energies, shape (n_samples,)."""
        encoder = self._build_encoder()
        node_features, edge_index, batch_vector = collate_graphs(graphs)
        with torch.no_grad():
            raw = encoder(node_features, edge_index, batch_vector)  # (n_samples, 1)
        return raw.squeeze(1).numpy() + self.bias


# ---------------------------------------------------------------------------
# Tuning spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProbeGnnTuningSpec:
    """Tune a ProbeGnnEncoder for direct adsorption energy regression via Optuna.

    Unlike the gate-based specs, this spec ignores ``dataset.mlip_features``
    entirely — all signal comes from the probe-augmented graph node features.
    """

    training_cfg: MoETrainingConfig
    hidden_dims: tuple[int, ...] = ()  # non-empty = fixed arch; empty = search

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
        in_features = train_graphs[0].node_features.shape[1]

        train_nf, train_ei, train_bv = collate_graphs(train_graphs)
        val_nf, val_ei, val_bv = collate_graphs(val_graphs)
        y_train = torch.tensor(train_ds.targets, dtype=torch.float32)
        y_val_np = val_ds.targets

        epochs = self.training_cfg.epochs
        seed = self.training_cfg.seed

        def objective(trial: Any) -> float:
            hidden_dim, n_layers = self._arch_from_trial(trial)
            lr: float = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            if seed is not None:
                torch.manual_seed(seed)
            encoder = ProbeGnnEncoder(in_features, hidden_dim, n_layers)
            _train_probe_encoder(
                encoder, train_nf, train_ei, train_bv, y_train,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
            )
            encoder.eval()
            with torch.no_grad():
                preds_val = encoder(val_nf, val_ei, val_bv).squeeze(1).numpy()
            return float(np.sqrt(np.mean((y_val_np - preds_val) ** 2)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> ProbeGnnModel:
        subsets = split.dataset_subsets()

        if refit_policy == "train_only":
            fit_graphs = _graphs_from_dataset(subsets.train)
            y_np = subsets.train.targets
        else:
            fit_graphs = (
                _graphs_from_dataset(subsets.train) + _graphs_from_dataset(subsets.val)
            )
            y_np = np.concatenate([subsets.train.targets, subsets.val.targets])

        in_features = fit_graphs[0].node_features.shape[1]
        hidden_dim, n_layers = self._arch_from_trial(best_trial)
        lr: float = best_trial.params["lr"]
        weight_decay: float = best_trial.params["weight_decay"]

        nf, ei, bv = collate_graphs(fit_graphs)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.training_cfg.seed is not None:
            torch.manual_seed(self.training_cfg.seed)
        encoder = ProbeGnnEncoder(in_features, hidden_dim, n_layers)
        _train_probe_encoder(
            encoder, nf, ei, bv, y,
            epochs=self.training_cfg.epochs, lr=lr, weight_decay=weight_decay,
        )

        encoder.eval()
        with torch.no_grad():
            preds_np = encoder(nf, ei, bv).squeeze(1).numpy()
        bias = float(np.mean(y_np - preds_np))

        return ProbeGnnModel(
            state_dict=encoder.state_dict(),
            in_features=in_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=bias,
        )

    def predict(self, model: ProbeGnnModel, dataset: SweepDataset) -> np.ndarray:
        return model.predict(_graphs_from_dataset(dataset))

    def trial_metadata(self, best_trial: Any, model: ProbeGnnModel) -> dict[str, Any]:
        return {
            "in_features": model.in_features,
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "bias": model.bias,
        }


# ---------------------------------------------------------------------------
# GNN direct baseline (same architecture, standard graph node features only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GnnDirectModel:
    """Frozen snapshot of a ProbeGnnEncoder trained on standard graph node features."""

    state_dict: dict[str, Any]
    in_features: int
    hidden_dim: int
    n_layers: int
    bias: float = 0.0

    def _build_encoder(self) -> ProbeGnnEncoder:
        encoder = ProbeGnnEncoder(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )
        encoder.load_state_dict(self.state_dict)
        encoder.eval()
        return encoder

    def predict(self, graphs: Sequence[GraphRecord]) -> np.ndarray:
        """Return predicted adsorption energies, shape (n_samples,)."""
        encoder = self._build_encoder()
        node_features, edge_index, batch_vector = collate_graphs(graphs)
        with torch.no_grad():
            raw = encoder(node_features, edge_index, batch_vector)
        return raw.squeeze(1).numpy() + self.bias


@dataclass(frozen=True, slots=True)
class GnnDirectTuningSpec:
    """Direct adsorption energy regression using standard graph node features.

    Identical architecture to ProbeGnnTuningSpec but reads from the dataset's
    regular graph_view rather than probe-augmented auxiliary records.
    """

    training_cfg: MoETrainingConfig
    hidden_dims: tuple[int, ...] = ()

    def _arch_from_trial(self, trial: Any) -> tuple[int, int]:
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

        train_graphs = _direct_graphs_from_dataset(train_ds)
        val_graphs = _direct_graphs_from_dataset(val_ds)
        in_features = train_graphs[0].node_features.shape[1]

        train_nf, train_ei, train_bv = collate_graphs(train_graphs)
        val_nf, val_ei, val_bv = collate_graphs(val_graphs)
        y_train = torch.tensor(train_ds.targets, dtype=torch.float32)
        y_val_np = val_ds.targets

        epochs = self.training_cfg.epochs
        seed = self.training_cfg.seed

        def objective(trial: Any) -> float:
            hidden_dim, n_layers = self._arch_from_trial(trial)
            lr: float = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            if seed is not None:
                torch.manual_seed(seed)
            encoder = ProbeGnnEncoder(in_features, hidden_dim, n_layers)
            _train_probe_encoder(
                encoder, train_nf, train_ei, train_bv, y_train,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
            )
            encoder.eval()
            with torch.no_grad():
                preds_val = encoder(val_nf, val_ei, val_bv).squeeze(1).numpy()
            return float(np.sqrt(np.mean((y_val_np - preds_val) ** 2)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> GnnDirectModel:
        subsets = split.dataset_subsets()

        if refit_policy == "train_only":
            fit_graphs = _direct_graphs_from_dataset(subsets.train)
            y_np = subsets.train.targets
        else:
            fit_graphs = (
                _direct_graphs_from_dataset(subsets.train)
                + _direct_graphs_from_dataset(subsets.val)
            )
            y_np = np.concatenate([subsets.train.targets, subsets.val.targets])

        in_features = fit_graphs[0].node_features.shape[1]
        hidden_dim, n_layers = self._arch_from_trial(best_trial)
        lr: float = best_trial.params["lr"]
        weight_decay: float = best_trial.params["weight_decay"]

        nf, ei, bv = collate_graphs(fit_graphs)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.training_cfg.seed is not None:
            torch.manual_seed(self.training_cfg.seed)
        encoder = ProbeGnnEncoder(in_features, hidden_dim, n_layers)
        _train_probe_encoder(
            encoder, nf, ei, bv, y,
            epochs=self.training_cfg.epochs, lr=lr, weight_decay=weight_decay,
        )

        encoder.eval()
        with torch.no_grad():
            preds_np = encoder(nf, ei, bv).squeeze(1).numpy()
        bias = float(np.mean(y_np - preds_np))

        return GnnDirectModel(
            state_dict=encoder.state_dict(),
            in_features=in_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=bias,
        )

    def predict(self, model: GnnDirectModel, dataset: SweepDataset) -> np.ndarray:
        return model.predict(_direct_graphs_from_dataset(dataset))

    def trial_metadata(self, best_trial: Any, model: GnnDirectModel) -> dict[str, Any]:
        return {
            "in_features": model.in_features,
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "bias": model.bias,
        }
