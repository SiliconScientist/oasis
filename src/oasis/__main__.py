import torch
from ase.io import read
from ase.optimize import BFGS
from ase.visualize import view
from ase.constraints import FixAtoms
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

from oasis.config import get_config
from oasis.processing import get_loaders
from oasis.model import get_model


def main():
    cfg = get_config()
    loaders = get_loaders(cfg)
    model = get_model(cfg)
    loss = 0.0
    for batch in loaders.test:
        x, y, atomic_reference = batch
        x = {k: v.to(cfg.trainer.device) for k, v in x.items()}
        y = y.to(cfg.trainer.device)
        atomic_reference = atomic_reference.to(cfg.trainer.device)
        slab_pred = model(x["slab"])
        ads_slab_pred = model(x["ads_slab"])
        ads_energy = ads_slab_pred - (slab_pred + atomic_reference)
        loss += torch.mean(torch.abs(ads_energy - y))
    print(loss / len(loaders.test))


if __name__ == "__main__":
    main()
