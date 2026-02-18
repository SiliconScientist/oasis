import re
import sympy as sp
from collections import Counter

from oasis.config import Config


def parse_formula(formula: str) -> dict[str, int]:
    """
    Convert chemical formula string -> {element: count}
    Works for simple formulas like CO2, H2O, CH3OH, O, etc.
    """
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    comp = Counter()
    for el, num in tokens:
        comp[el] += int(num) if num else 1
    return dict(comp)


def build_basis_matrix(cfg) -> sp.Matrix:
    """
    Build A (elements x basis_species) for arbitrary basis.

    Uses:
      cfg.ingest.stoich.elements       e.g. ["C","H","O","N"]
      cfg.ingest.stoich.basis_species  e.g. ["CO2","H2O","H2","NH3"]
    """
    elements = list(cfg.ingest.stoich.elements)
    species = list(cfg.ingest.stoich.basis_species)

    rows = []
    for el in elements:
        row = []
        for spc in species:
            comp = parse_formula(spc)
            row.append(comp.get(el, 0))
        rows.append(row)

    return sp.Matrix(rows)


def build_b_vector(
    elements: list[str], target_composition: dict[str, int]
) -> sp.Matrix:
    """
    Build b (elements x 1) from a dict like {"C":1,"H":4,"O":1}.
    """
    return sp.Matrix([int(target_composition.get(el, 0)) for el in elements])


def solve_stoichiometry(cfg: Config, target_composition: dict[str, int]) -> list[int]:
    A = build_basis_matrix(cfg)
    elements = list(cfg.ingest.stoich.elements)
    b = build_b_vector(elements, target_composition)
    x = A.LUsolve(b)
    return [float(v) for v in list(x)]
