import os
import polars as pl
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
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from mattersim.forcefield import MatterSimCalculator
from fairchem.core import FAIRChemCalculator
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from oasis.config import Config


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


def get_adsorption_predictions(
    ads_slab_list: list[Atoms], slab_list: list[Atoms], calc_dict: dict[str, Calculator]
):
    mlip_predictions = {}
    for name, calc in calc_dict.items():
        predictions = []
        for ads_slab, slab in zip(ads_slab_list, slab_list):
            ads_slab.calc = calc
            slab.calc = calc
            slab_energy = slab.get_potential_energy()
            ads_slab_energy = ads_slab.get_potential_energy()
            adsorbate_reference_energy = ads_slab.info["adsorbate_reference_energy"]
            prediction = ads_slab_energy - slab_energy - adsorbate_reference_energy
            predictions.append(prediction)
        mlip_predictions[name] = predictions
    df = pl.DataFrame(mlip_predictions)
    return df


def get_relaxed_systems(cfg: Config):
    if not (
        os.path.exists(cfg.data.relaxed_slabs)
        and os.path.exists(cfg.data.relaxed_systems)
    ):
        if cfg.dev_run:
            ads_slab_list = read(cfg.data.ideal_systems, index=":5")
        else:
            ads_slab_list = read(cfg.data.ideal_systems, index=":")
        slab_list = trim_tagged_atoms(
            atoms_list=ads_slab_list, tag=cfg.processing.adsorbate_tag
        )
        for slab in ads_slab_list:
            slab.info["y_ads"] = slab.get_potential_energy()
            slab.info["adsorbate_reference_energy"] = (
                compute_adsorbate_reference_energy(slab, cfg.processing.adsorbate_tag)
            )
        index_fn = partial(indices_from_tags, tags=cfg.processing.constrained_tags)
        slab_list = constrain_atoms(atoms_list=slab_list, index_fn=index_fn)
        ads_slab_list = constrain_atoms(atoms_list=ads_slab_list, index_fn=index_fn)
        calc = mace_mp(
            model=cfg.models.mace.checkpoint,
            default_dtype=cfg.models.dtype,
            device=cfg.models.device,
            head=cfg.models.mace.head,
        )
        relaxed_slabs = batch_relax(slab_list, calc)
        relaxed_ads_slabs = batch_relax(ads_slab_list, calc)
        write(
            filename=cfg.data.relaxed_slabs,
            images=relaxed_slabs,
        )
        write(
            filename=cfg.data.relaxed_systems,
            images=relaxed_ads_slabs,
        )
        return relaxed_slabs, relaxed_ads_slabs
    else:
        relaxed_slabs = read(filename=cfg.data.relaxed_slabs, index=":")
        relaxed_ads_slabs = read(filename=cfg.data.relaxed_systems, index=":")
        return relaxed_slabs, relaxed_ads_slabs


def build_calculators(cfg):
    device = cfg.models.device
    dtype = cfg.models.dtype

    # --- MACE ---
    mace_cfg = cfg.models.mace
    mace_calc = mace_mp(
        model=mace_cfg.checkpoint,
        default_dtype=dtype,
        device=device,
        head=mace_cfg.head,
    )

    # --- Orb ---
    orb_cfg = cfg.models.orb
    orbff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision=orb_cfg.precision,
    )
    orb_calc = ORBCalculator(orbff, device=device)

    # --- MatterSim ---
    ms_cfg = cfg.models.mattersim
    mattersim_calc = MatterSimCalculator(
        load_path=str(ms_cfg.checkpoint),
        device=device,
    )

    # --- UMA / FairChem ---
    uma_cfg = cfg.models.uma
    predictor = load_predict_unit(uma_cfg.checkpoint, device=device)
    uma_calc = FAIRChemCalculator(predictor, task_name=uma_cfg.task)

    # --- CHGNet ---
    chgnet = CHGNet.load()
    chgnet_calc = CHGNetCalculator(chgnet, use_device=device)

    calc_dict = {
        "mace": mace_calc,
        "orb": orb_calc,
        "mattersim": mattersim_calc,
        "uma": uma_calc,
        "chgnet": chgnet_calc,
    }

    return calc_dict


def get_data(cfg):
    relaxed_slabs, relaxed_ads_slabs = get_relaxed_systems(cfg)
    y_labels = [ads_slab.info["y_ads"] for ads_slab in relaxed_ads_slabs]
    calc_dict = build_calculators(cfg)
    pred_df = get_adsorption_predictions(relaxed_ads_slabs, relaxed_slabs, calc_dict)
    df = pl.DataFrame({"y_true": y_labels}).with_columns(pred_df)
    return df
