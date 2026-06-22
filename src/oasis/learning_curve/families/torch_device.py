from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def resolve_torch_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device


def tensors_to_device(device: torch.device, *tensors: Tensor) -> tuple[Tensor, ...]:
    return tuple(tensor.to(device) for tensor in tensors)


def tensor_to_numpy(tensor: Tensor):
    return tensor.detach().cpu().numpy()


def state_dict_to_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in state_dict.items()}
