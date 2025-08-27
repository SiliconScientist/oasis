import os
from typing import Callable
import torch
from ase.io import read, write
from ase.atoms import Atoms
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from ase.optimize import BFGS
from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch


ATOMIC_REFERENCE_ENERGIES = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}


def collate_atomic_pairs(batch):
    # batch: list of (system_atomic, surface_atomic, y_ads)
    slab_list, ads_slab_list, y_ads, atomic_references = zip(*batch)
    x = {
        "slab": atomicdata_list_to_batch(list(slab_list)),
        "ads_slab": atomicdata_list_to_batch(list(ads_slab_list)),
    }
    y = torch.as_tensor(y_ads, dtype=torch.float32)  # shape [B]
    atomic_reference = torch.as_tensor(
        atomic_references, dtype=torch.float32
    )  # shape [B]
    return x, y, atomic_reference


@dataclass
class DataLoaderSplits:
    holdout: DataLoader
    test: DataLoader


def indice_from_tags(atoms: Atoms, tags: list[int]):
    return [i for i, atom in enumerate(atoms) if atom.tag in tags]


def constrain_atoms(atoms_list: list[Atoms], index_fn: Callable) -> list[Atoms]:
    constrained_atoms_list = []
    for atoms in atoms_list:
        atoms.pbc = True
        indices = index_fn(atoms)
        atoms.set_constraint(FixAtoms(indices=indices))
        constrained_atoms_list.append(atoms)
    return constrained_atoms_list


def batch_relax(
    atoms_list: list[Atoms], calc: Calculator, fmax: float = 0.05, steps: int = 100
) -> list[Atoms]:
    relaxed_atoms_list = []
    for atoms in atoms_list:
        atoms.calc = calc
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        relaxed_atoms_list.append(atoms)
    return relaxed_atoms_list


def get_loaders(cfg):
    slab_list = read(cfg.data.raw.slabs, index=":")
    ads_slab_list = read(cfg.data.raw.ads_slabs, index=":")
    y_labels = [ads_slab.get_potential_energy() for ads_slab in ads_slab_list]
    if not (
        os.path.exists(cfg.data.raw_relaxed.slabs)
        and os.path.exists(cfg.data.raw_relaxed.ads_slabs)
    ):
        index_fn = partial(indice_from_tags, tags=[0])
        slab_list = constrain_atoms(atoms_list=slab_list, index_fn=index_fn)
        ads_slab_list = constrain_atoms(atoms_list=ads_slab_list, index_fn=index_fn)
        predict_unit = load_predict_unit(
            path=cfg.model.checkpoint,
            device=cfg.trainer.device,
        )
        calc = FAIRChemCalculator(predict_unit, task_name="oc20")
        relaxed_slabs = batch_relax(slab_list, calc)
        relaxed_ads_slabs = batch_relax(ads_slab_list, calc)
        write(
            filename=cfg.data.raw_relaxed.slabs,
            images=relaxed_slabs,
        )
        write(
            filename=cfg.data.raw_relaxed.ads_slabs,
            images=relaxed_ads_slabs,
        )
    else:
        relaxed_slabs = read(filename=cfg.data.raw_relaxed.slabs, index=":")
        relaxed_ads_slabs = read(filename=cfg.data.raw_relaxed.ads_slabs, index=":")

    if not (os.path.exists(cfg.data.holdout) and os.path.exists(cfg.data.test)):
        atomic_data_list = []
        process_fn = partial(
            AtomicData.from_ase, task_name="oc20", r_edges=True, max_neigh=45
        )
        for slab, ads_slab, y in zip(relaxed_slabs, relaxed_ads_slabs, y_labels):
            slab_data = process_fn(input_atoms=slab)
            ads_slab_data = process_fn(input_atoms=ads_slab)
            atomic_reference = sum(
                [
                    ATOMIC_REFERENCE_ENERGIES[x]
                    for x in ads_slab[ads_slab.get_tags() == 2].get_chemical_symbols()
                ]
            )
            atomic_data_list.append([slab_data, ads_slab_data, y, atomic_reference])
        holdout, test = train_test_split(
            atomic_data_list, train_size=10, random_state=cfg.seed
        )
        torch.save(holdout, cfg.data.holdout)
        torch.save(test, cfg.data.test)
    else:
        holdout = torch.load(cfg.data.holdout)
        test = torch.load(cfg.data.test)
    dataloader = partial(
        DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_atomic_pairs
    )
    loaders = DataLoaderSplits(
        holdout=dataloader(holdout, shuffle=True),
        test=dataloader(test, shuffle=False),
    )
    return loaders
