import sympy as sp

from oasis.config import Config


def build_basis_matrix(cfg) -> sp.Matrix:
    """
    Build A (elements x basis_species) from:
      cfg.stoich.elements         e.g. ["C","H","O"]
      cfg.stoich.basis_species    e.g. ["CO2","H2O","H2"]
      cfg.stoich.basis_composition (dict) mapping species -> {element: count}
    """
    elements = list(cfg.ingest.stoich.elements)
    species = list(cfg.ingest.stoich.basis_species)
    comp = cfg.ingest.stoich.basis_composition  # dict-like

    # A[i,j] = count of element i in species j
    rows = []
    for el in elements:
        row = []
        for spc in species:
            # tolerate missing keys (e.g., H2 has no O entry)
            row.append(int(comp.get(spc, {}).get(el, 0)))
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
