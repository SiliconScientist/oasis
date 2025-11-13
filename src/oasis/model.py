import hydra
import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.utils import load_inference_model
# from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
# from mattersim.forcefield.potential import batch_to_dict, Potential
# from torch_geometric.data import Data

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


# class MS(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = Potential(model)

#     def forward(self, data: Data) -> torch.Tensor:
#         data = batch_to_dict(data)
#         output = self.model(data, include_forces=False, include_stresses=False)
#         energies = output["total_energy"]  # eV
#         return energies


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
    model = UMA(model, tasks)

    # checkpoint = torch.load(
    #     f=cfg.model.checkpoint,
    #     map_location=torch.device(cfg.trainer.device),
    #     weights_only=False,
    # )
    # model = M3Gnet(**checkpoint["model_args"])
    # model.load_state_dict(checkpoint["model"], strict=False)
    # model.to(cfg.trainer.device)
    # model = MS(model)

    return model
