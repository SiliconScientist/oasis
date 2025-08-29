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

    # slab = read(cfg.data.surfaces, index=2)
    # ads_slab = read(cfg.data.systems, index=2)
    # y_label = ads_slab.get_potential_energy()
    # slab.pbc = True
    # ads_slab.pbc = True
    # slab.set_constraint(FixAtoms(mask=[atom.tag < 1 for atom in slab]))
    # ads_slab.set_constraint(FixAtoms(mask=[atom.tag < 1 for atom in ads_slab]))
    # predict_unit = load_predict_unit(
    #     path=cfg.model.checkpoint,
    #     device=cfg.trainer.device,
    # )
    # calc = FAIRChemCalculator(
    #     predict_unit=predict_unit,
    #     task_name="oc20",
    # )
    # slab.calc = calc
    # opt = BFGS(slab)
    # opt.run(fmax=0.05, steps=100)
    # slab_e = slab.get_potential_energy()

    # ads_slab.calc = calc
    # opt = BFGS(ads_slab)
    # opt.run(fmax=0.05, steps=100)
    # ads_slab_e = ads_slab.get_potential_energy()
    # atomic_reference_energies = {
    #     "H": -3.477,
    #     "N": -8.083,
    #     "O": -7.204,
    #     "C": -7.282,
    # }
    # # Energy for ((H2O-H2) + * -> *O) + (H2 + 1/2O2 -> H2O) leads to 1/2O2 + * -> *O!
    # ads_energy = (
    #     ads_slab_e
    #     - slab_e
    #     - sum(
    #         [
    #             atomic_reference_energies[x]
    #             for x in ads_slab[ads_slab.get_tags() == 2].get_chemical_symbols()
    #         ]
    #     )
    # )
    # mae = torch.abs(torch.tensor(ads_energy) - torch.tensor(y_label))
    # print(mae)
    loaders = get_loaders(cfg)
    model = get_model(cfg)
    loss = 0.0
    for batch in loaders.test:
        x, y, atomic_reference = batch
        slab_pred = model(x["slab"])
        ads_slab_pred = model(x["ads_slab"])
        ads_energy = ads_slab_pred - (slab_pred + atomic_reference)
        loss += torch.mean(torch.abs(ads_energy - y))
        print("Pause")
    print(loss / len(loaders.test))


if __name__ == "__main__":
    main()
