from ase.io import read
from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch


@dataclass
class DataLoaders:
    holdout: DataLoader
    test: DataLoader


def get_loaders(cfg):
    atoms_list = read(filename=cfg.data.raw, index=":")
    atomic_data_list = []
    process_fn = partial(
        AtomicData.from_ase, task_name="oc20", r_edges=True, max_neigh=45
    )
    for atoms in atoms_list:
        atomic_data = process_fn(input_atoms=atoms)
        atomic_data_list.append(atomic_data)
    holdout, test = train_test_split(
        atomic_data_list, train_size=10, random_state=cfg.seed
    )
    dataloader = partial(
        DataLoader, **cfg.dataloader.model_dump(), collate_fn=atomicdata_list_to_batch
    )
    loaders = DataLoaders(
        holdout=dataloader(holdout, shuffle=True), test=dataloader(test, shuffle=False)
    )
    return loaders
