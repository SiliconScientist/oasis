import hydra
import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.utils import load_inference_model

from oasis.config import Config


class UMA(nn.Module):
    def __init__(self, model, tasks):
        super().__init__()
        self.model: nn.Module = model
        self.tasks = {t.name: t for t in tasks}

    def forward(self, data: AtomicData) -> torch.Tensor:
        output = self.model(data)
        pred_output = {}
        for task_name, task in self.tasks.items():
            pred_output[task_name] = task.normalizer.denorm(
                output[task_name][task.property]
            )
            if task.element_references is not None:
                pred_output[task_name] = task.element_references.undo_refs(
                    data, pred_output[task_name]
                )
        return pred_output["oc20_energy"].float()


def get_model(cfg: Config):
    model, checkpoint = load_inference_model(
        checkpoint_location=cfg.model.checkpoint,
        return_checkpoint=True,
    )
    model = model.to(cfg.trainer.device)
    tasks = [
        hydra.utils.instantiate(task_config) for task_config in checkpoint.tasks_config
    ]
    for task in tasks:
        if task.element_references is not None:
            task.element_references.to(cfg.trainer.device)
    return UMA(model, tasks=tasks)
