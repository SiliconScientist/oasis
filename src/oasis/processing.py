import os
from typing import Callable
import torch
from ase.visualize import view
from ase.io import read, write
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from mace.calculators import mace_mp
from ase.optimize import BFGS
from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


ATOMIC_REFERENCE_ENERGIES = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}


def collate_atomic_pairs(batch, collate_fn: Callable = None):
    # batch: list of (system_atomic, surface_atomic, y_ads)
    slab_list, ads_slab_list, y_ads, atomic_references = zip(*batch)
    x = {
        "slab": collate_fn(list(slab_list)),
        "ads_slab": collate_fn(list(ads_slab_list)),
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


def indices_from_tags(atoms: Atoms, tags: list[int]):
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


def compute_adsorbate_reference_energy(slab, tag):
    mask = slab.get_tags() == tag
    symbols = slab[mask].get_chemical_symbols()
    return sum(ATOMIC_REFERENCE_ENERGIES[s] for s in symbols)


def trim_tagged_atoms(atoms_list: list[Atoms], tag: int):
    trimmed_atoms_list = []
    for atoms in atoms_list:
        atoms = atoms.copy()
        del atoms[atoms.get_tags() == tag]
        trimmed_atoms_list.append(atoms)
    return trimmed_atoms_list


def get_data(cfg):
    ads_slab_list = read(cfg.data.ideal_systems, index=":1")
    slab_list = trim_tagged_atoms(
        atoms_list=ads_slab_list, tag=cfg.processing.adsorbate_tag
    )
    y_labels = [slab.get_potential_energy() for slab in ads_slab_list]

    for slab in ads_slab_list:
        slab.info["adsorbate_reference_energy"] = compute_adsorbate_reference_energy(
            slab, cfg.processing.adsorbate_tag
        )
    if not (
        os.path.exists(cfg.data.relaxed_systems)
        and os.path.exists(cfg.data.relaxed_slabs)
    ):
        index_fn = partial(indices_from_tags, tags=cfg.processing.constrained_tags)
        slab_list = constrain_atoms(atoms_list=slab_list, index_fn=index_fn)
        ads_slab_list = constrain_atoms(atoms_list=ads_slab_list, index_fn=index_fn)
        calc = mace_mp(
            model=cfg.model.checkpoint,
            default_dtype="float64",
            device="cpu",
            head="omat_pbe",
        )
        relaxed_slabs = batch_relax(slab_list, calc)
        relaxed_ads_slabs = batch_relax(ads_slab_list, calc)
        write(
            filename=cfg.data.relaxed_systems,
            images=relaxed_ads_slabs,
        )
        write(
            filename=cfg.data.relaxed_slabs,
            images=relaxed_slabs,
        )
    else:
        relaxed_slabs = read(filename=cfg.data.relaxed_systems, index=":")
        relaxed_ads_slabs = read(filename=cfg.data.relaxed_systems, index=":")
    return relaxed_slabs, relaxed_ads_slabs, y_labels
