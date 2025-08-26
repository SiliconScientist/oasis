import torch
from ase.io import read
from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch


def collate_atomic_pairs(batch):
    # batch: list of (system_atomic, surface_atomic, y_ads)
    sys_list, surf_list, y_ads = zip(*batch)
    x = {
        "system": atomicdata_list_to_batch(list(sys_list)),
        "surface": atomicdata_list_to_batch(list(surf_list)),
    }
    y = torch.as_tensor(y_ads, dtype=torch.float32)  # shape [B]
    return x, y


@dataclass
class DataLoaders:
    holdout: DataLoader
    test: DataLoader


def get_loaders(cfg):
    systems_list = read(filename=cfg.data.systems, index=":")
    surface_list = read(filename=cfg.data.surfaces, index=":")
    atomic_data_list = []
    process_fn = partial(
        AtomicData.from_ase, task_name="oc20", r_edges=True, max_neigh=45
    )
    for system, surface in zip(systems_list, surface_list):
        system_data = process_fn(input_atoms=system)
        surface_data = process_fn(input_atoms=surface)
        atomic_data_list.append([system_data, surface_data, system_data.energy])
    holdout, test = train_test_split(
        atomic_data_list, train_size=10, random_state=cfg.seed
    )
    dataloader = partial(
        DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_atomic_pairs
    )
    loaders = DataLoaders(
        holdout=dataloader(holdout, shuffle=True),
        test=dataloader(test, shuffle=False),
    )
    return loaders
