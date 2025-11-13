from typing import Callable
from ase.io import read, write
from ase.visualize import view
from functools import partial
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator


def indice_from_tags(atoms: Atoms, tags: list[int]):
    return [i for i, atom in enumerate(atoms) if atom.tag in tags]


def indices_below_height(atoms: Atoms, height: float):
    positions = atoms.get_positions()
    return [i for i, pos in enumerate(positions) if pos[2] < height]


def constrain_atoms(atoms_list: list[Atoms], index_fn: Callable) -> list[Atoms]:
    constrained_atoms_list = []
    for atoms in atoms_list:
        new_atoms = atoms.copy()
        new_atoms.pbc = True
        indices = index_fn(atoms)
        new_atoms.set_constraint(FixAtoms(indices=indices))
        constrained_atoms_list.append(new_atoms)
    return constrained_atoms_list


def batch_relax(
    atoms_list: list[Atoms], calc: Calculator, fmax: float = 0.05, steps: int = 100
) -> list[Atoms]:
    relaxed_atoms_list = []
    for atoms in atoms_list:
        new_atoms = atoms.copy()
        new_atoms.calc = calc
        opt = BFGS(new_atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        relaxed_atoms_list.append(new_atoms)
    return relaxed_atoms_list


atoms_list = read("data/raw/catalonia.traj", index=":5")
energy_list = [
    atoms.get_potential_energy() if getattr(atoms, "calc", None) is not None else None
    for atoms in atoms_list
]
# index_fn = partial(indice_from_tags, tags=[0])
# index_fn = partial(indices_below_height, height=14.0)
# atoms_list = constrain_atoms(atoms_list=atoms_list, index_fn=index_fn)
predict_unit = load_predict_unit(
    path="/Users/averyhill/github/oasis/data/checkpoints/uma-s-1p1.pt", device="cpu"
)
calc = FAIRChemCalculator(predict_unit=predict_unit, task_name="oc20")
relaxed_atoms_list = batch_relax(atoms_list=atoms_list, calc=calc)
images = []
for atoms, energy in zip(relaxed_atoms_list, energy_list):
    atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy)
    images.append(atoms)
write(filename="uma_relaxed_catalonia.traj", images=images)
