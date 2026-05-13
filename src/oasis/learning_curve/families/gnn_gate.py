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
from oasis.learning_curve.families.gating_policy import DenseGatingPolicy, GatingPolicy
from oasis.sweep import GraphRecord, SweepDataset, TrainValTestSweepRunnerInput
from oasis.tune import SelectionRefitPolicy


def collate_graphs(graphs: Sequence[GraphRecord]) -> tuple[Tensor, Tensor, Tensor]:
    """Batch a sequence of GraphRecords into sparse tensors.

    Returns:
        node_features: (total_nodes, n_features) float32
        edge_index:    (2, total_edges) long, with per-graph node offsets applied
        batch_vector:  (total_nodes,) long, graph index for each node
    """
    all_node_features: list[Tensor] = []
    all_edge_index: list[Tensor] = []
    all_batch: list[Tensor] = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        n = graph.n_nodes
        all_node_features.append(
            torch.tensor(graph.node_features, dtype=torch.float32)
        )
        ei = torch.tensor(graph.edge_index, dtype=torch.long) + node_offset
        all_edge_index.append(ei)
        all_batch.append(torch.full((n,), graph_idx, dtype=torch.long))
        node_offset += n

    node_features = torch.cat(all_node_features, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros((2, 0), dtype=torch.long)
    batch_vector = torch.cat(all_batch, dim=0)

    return node_features, edge_index, batch_vector


def _scatter_mean_nodes(
    src_features: Tensor,
    dst_indices: Tensor,
    n_nodes: int,
) -> Tensor:
    """Mean-aggregate src_features into destination node buckets."""
    hidden = src_features.shape[1]
    out = torch.zeros(n_nodes, hidden, dtype=src_features.dtype, device=src_features.device)
    if src_features.shape[0] == 0:
        return out
    idx = dst_indices.unsqueeze(1).expand_as(src_features)
    out.scatter_add_(0, idx, src_features)
    count = torch.zeros(n_nodes, 1, dtype=src_features.dtype, device=src_features.device)
    count.scatter_add_(0, dst_indices.unsqueeze(1), torch.ones(dst_indices.shape[0], 1, dtype=src_features.dtype, device=src_features.device))
    return out / count.clamp(min=1)


def _global_mean_pool(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    """Mean-pool node features into per-graph vectors."""
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
    count.scatter_add_(0, batch.unsqueeze(1), torch.ones(batch.shape[0], 1, dtype=x.dtype, device=x.device))
    return out / count.clamp(min=1)


class GnnEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.mp_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_vector: Tensor,
    ) -> Tensor:
        n_nodes = node_features.shape[0]
        n_graphs = int(batch_vector.max().item()) + 1

        h = self.input_proj(node_features)  # (n_nodes, hidden_dim)

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        for layer in self.mp_linears:
            agg = _scatter_mean_nodes(h[edge_src], edge_dst, n_nodes)
            h = F.relu(layer(h + agg))

        pooled = _global_mean_pool(h, batch_vector, n_graphs)  # (n_graphs, hidden_dim)
        return self.output_proj(pooled)  # (n_graphs, out_features)


@dataclass(frozen=True, slots=True)
class GnnGateModel:
    state_dict: dict[str, Any]
    n_experts: int
    hidden_dim: int
    n_layers: int
    bias: float = 0.0
    policy: GatingPolicy = field(default_factory=DenseGatingPolicy)

    def _build_encoder(self, in_features: int) -> GnnEncoder:
        encoder = GnnEncoder(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            out_features=self.n_experts,
            n_layers=self.n_layers,
        )
        encoder.load_state_dict(self.state_dict)
        encoder.eval()
        return encoder

    def predict(self, X: np.ndarray, graphs: Sequence[GraphRecord]) -> np.ndarray:
        in_features = graphs[0].node_features.shape[1]
        encoder = self._build_encoder(in_features)
        node_features, edge_index, batch_vector = collate_graphs(graphs)
        with torch.no_grad():
            logits = encoder(node_features, edge_index, batch_vector)  # (n_samples, n_experts)
        weights = self.policy.apply(logits.numpy())  # (n_samples, n_experts)
        return (X * weights).sum(axis=1) + self.bias


def _graphs_from_dataset(dataset: SweepDataset) -> list[GraphRecord]:
    return [dataset.graphs[sid] for sid in dataset.sample_ids.tolist()]


def _train_encoder(
    encoder: GnnEncoder,
    node_features: Tensor,
    edge_index: Tensor,
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
        logits = encoder(node_features, edge_index, batch_vector)
        weights = torch.softmax(logits, dim=1)
        preds = (X * weights).sum(dim=1)
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.step()


@dataclass(frozen=True, slots=True)
class GnnGateTuningSpec:
    training_cfg: MoETrainingConfig
    hidden_dims: tuple[int, ...] = ()
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
        in_features = train_graphs[0].node_features.shape[1]
        n_experts = train_ds.mlip_features.shape[1]

        train_nf, train_ei, train_bv = collate_graphs(train_graphs)
        val_nf, val_ei, val_bv = collate_graphs(val_graphs)
        X_train = torch.tensor(train_ds.mlip_features, dtype=torch.float32)
        y_train = torch.tensor(train_ds.targets, dtype=torch.float32)
        X_val_np = val_ds.mlip_features
        y_val_np = val_ds.targets

        epochs = self.training_cfg.epochs
        seed = self.training_cfg.seed
        policy = self.policy

        def objective(trial: Any) -> float:
            hidden_dim, n_layers = self._arch_from_trial(trial)
            lr: float = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay: float = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            if seed is not None:
                torch.manual_seed(seed)
            encoder = GnnEncoder(in_features, hidden_dim, n_experts, n_layers)
            _train_encoder(
                encoder, train_nf, train_ei, train_bv, X_train, y_train,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
            )

            encoder.eval()
            with torch.no_grad():
                logits = encoder(val_nf, val_ei, val_bv)
            weights = policy.apply(logits.numpy())
            preds = (X_val_np * weights).sum(axis=1)
            return float(np.sqrt(np.mean((y_val_np - preds) ** 2)))

        return objective

    def fit_selected_model(
        self,
        split: TrainValTestSweepRunnerInput,
        best_trial: Any,
        *,
        refit_policy: SelectionRefitPolicy,
    ) -> GnnGateModel:
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

        in_features = fit_graphs[0].node_features.shape[1]
        n_experts = split.dataset.mlip_features.shape[1]
        hidden_dim, n_layers = self._arch_from_trial(best_trial)
        lr: float = best_trial.params["lr"]
        weight_decay: float = best_trial.params["weight_decay"]

        nf, ei, bv = collate_graphs(fit_graphs)
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.training_cfg.seed is not None:
            torch.manual_seed(self.training_cfg.seed)
        encoder = GnnEncoder(in_features, hidden_dim, n_experts, n_layers)
        _train_encoder(
            encoder, nf, ei, bv, X, y,
            epochs=self.training_cfg.epochs, lr=lr, weight_decay=weight_decay,
        )

        encoder.eval()
        with torch.no_grad():
            logits = encoder(nf, ei, bv)
        weights = self.policy.apply(logits.numpy())
        preds = (X_np * weights).sum(axis=1)
        bias = float(np.mean(y_np - preds))

        return GnnGateModel(
            state_dict=encoder.state_dict(),
            n_experts=n_experts,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=bias,
            policy=self.policy,
        )

    def predict(self, model: GnnGateModel, dataset: SweepDataset) -> np.ndarray:
        return model.predict(dataset.mlip_features, _graphs_from_dataset(dataset))

    def trial_metadata(self, best_trial: Any, model: GnnGateModel) -> dict[str, Any]:
        return {
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "n_experts": model.n_experts,
            "bias": model.bias,
        }
