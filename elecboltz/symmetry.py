import re
import numpy as np
from numpy.typing import NDArray


ALIASES = {"cubic": "Oh", "orthorhombic": "D2h", "tetragonal": "D4h",
           "trigonal": "D3d", "hexagonal": "D6h"}


class Symmetry:
    """Class representing the point group symmetry
    of a crystal structure.

    Parameters
    ----------
    name
        Name of the symmetry group. Can be the lattice type (e.g.,
        'tetragonal', 'trigonal', 'hexagonal'), or a symmetry symbol:
        a Schoenflies symbol (e.g., 'D4h', 'D3d', 'C4v')
        or a Hermann-Mauguin symbol (e.g., '4/mmm', '-3m').
    tol
        Tolerance for floating-point comparisons. Used to determine
        the distinctness of points, symmetry operations, and planes.
    Attributes
    ----------
    matrices : list[NDArray[np.floating]]
        List of 3x3 orthogonal matrices representing the symmetry
        operations.
    wedge_planes : list[NDArray[np.floating]]
        List of normal vectors of the irreducible wedge planes.
    """
    def __init__(self, name: str, tol=1e-5):
        self.name = name
        self.tol = tol
        self.matrices = get_point_group_matrices(name, tol=tol)
        self.wedge_planes = get_irreducible_wedge_planes(name)


def get_point_group_matrices(name: str, tol: float=1e-5
                             ) -> list[NDArray[np.floating]]:
    """Get the point group symmetry operations.

    Parameters
    ----------
    name
        Name of the symmetry group. Can be the lattice type (e.g.,
        'tetragonal', 'trigonal', 'hexagonal'), or a symmetry symbol:
        a Schoenflies symbol (e.g., 'D4h', 'D3d', 'C4v')
        or a Hermann-Mauguin symbol (e.g., '4/mmm', '-3m').
    tol
        Tolerance for floating-point comparisons. Used to distinguish
        unique symmetry operations.

    Returns
    -------
    list[NDArray[np.floating]]
        List of 3x3 orthogonal matrices representing the symmetry
        operations of the specified point group.
    """
    family, n, suffix = _get_sym_info(name)
    generators = []
    # Handle special low-symmetry symbols
    if family == "Ci":
        generators.append(INVERSION)
    elif family == "Cs":
        generators.append(SIGMA_H)
    elif family in ("T", "O"): # Cubic groups
        _add_cubic_generators(generators, family, suffix)
    else: # Axial groups (C_n, D_n, S_n)
        _add_axial_generators(generators, family, n, suffix)
    return _close_group(generators, tol=tol)


def get_irreducible_wedge_planes(name: str, tol: float = 1e-5
                                 ) -> list[NDArray[np.floating]]:
    """Extracts unit normal vectors of the planes bounding the
    irreducible wedge of a crystal structure's Brillouin zone.

    Parameters
    ----------
        name
            Name of the symmetry group. Can be the lattice type (e.g.,
            'tetragonal', 'trigonal', 'hexagonal'), or a symmetry symbol:
            a Schoenflies symbol (e.g., 'D4h', 'D3d', 'C4v')
            or a Hermann-Mauguin symbol (e.g., '4/mmm', '-3m').
        tol
            Tolerance for floating-point comparisons. Used to determine
            the distinctness of plane normals.

    Returns
    -------
    list[NDArray[np.floating]]
        List of 3D vectors representing the bounding plane normals.
        Each normal n defines a bounding plane n . k = 0, oriented
        such that points inside the wedge satisfy n . k >= 0.
    """
    family, n, suffix = _get_sym_info(name)
    planes = []

    if family in ("Ci", "Cs"): # Inversion-only / Horizontal mirror-only
        planes.append(np.array([0.0, 0.0, 1.0]))  # kz = 0
    elif family in ("T", "O"): # Cubic Groups (O, Oh, T, Td, Th)
        # Canonical 1/48th cubic wedge: kx >= ky >= kz >= 0
        planes.append(np.array([1.0, -1.0, 0.0]))  # kx = ky
        planes.append(np.array([0.0, 1.0, -1.0]))  # ky = kz
        planes.append(np.array([0.0, 0.0, 1.0]))   # kz = 0
    else: # Axial Groups (C_n, D_n, S_n)
        # Determine the in-plane wedge angle Delta_theta
        # Dihedral groups (D_n) and vertical/dihedral mirrors (C_nv, D_nd)
        # bisect the n-fold sector.
        has_inplane_bisector = (family == "D") or (suffix in ("v", "d"))
        wedge_angle = np.pi / n if has_inplane_bisector else 2 * np.pi / n

        if n > 1 or has_inplane_bisector:
            # Plane 1: ky = 0 (phi = 0 boundary)
            planes.append(np.array([0.0, 1.0, 0.0]))
            # Plane 2: phi = wedge_angle boundary
            # Inside the wedge: kx * sin(theta) - ky * cos(theta) >= 0
            planes.append(np.array(
                [np.sin(wedge_angle), -np.cos(wedge_angle), 0.0]))
        # Out-of-plane kz = 0 plane:
        # Included when the group contains an operation mapping kz -> -kz:
        # - sigma_h (suffix 'h' or family 'Cs')
        # - 2-fold in-plane axes (family 'D')
        # - Inversion / improper rotoreflections (suffix 'd', 'i', or family 'S')
        if suffix in ("h", "d", "i", "j") or family in ("D", "S"):
            planes.append(np.array([0.0, 0.0, 1.0]))

    planes = [normal / np.linalg.norm(normal) for normal in planes]
    return _remove_deduplicate_planes(planes, tol=tol)


def _remove_deduplicate_planes(planes, tol=1e-5):
    unique = []
    for plane in planes:
        plane = np.asarray(plane, dtype=float)
        if plane.size != 3:
            continue
        norm = np.linalg.norm(plane)
        if np.isclose(norm, 0.0, atol=tol):
            continue
        plane = plane / norm
        if not any(np.allclose(plane, other, atol=tol)
                   or np.allclose(np.abs(plane), np.abs(other), atol=tol)
                   for other in unique):
            unique.append(plane)
    return unique


def _add_cubic_generators(generators, family, suffix):
    # Tetrahedral core rotations
    generators.extend([_rot_z(2), C2_X, ROT_C3_111])
    if family == "T":
        if suffix == "h":
            generators.append(INVERSION)
        elif suffix == "d":
            generators.append(SIGMA_D)
    elif family == "O":
        generators.append(_rot_z(4))
        if suffix == "h":
            generators.append(INVERSION)


def _add_axial_generators(generators, family, n, suffix):
    if n < 1:
        raise ValueError(f"Invalid symmetry group: {family}{n}{suffix}")
    if family == "S": # Improper rotation groups
        generators.append(_rot_z(n) @ SIGMA_H)
    else: # Other axial groups (C_n, D_n)
        if n > 1:
            generators.append(_rot_z(n))
        if family == "D":
            generators.append(C2_X)

        if suffix == "h":
            generators.append(SIGMA_H)
        elif suffix == "v":
            generators.append(SIGMA_V)
        elif suffix == "d":
            if n % 2 == 1:
                generators.append(INVERSION)  # D3d = D3 x {E, i}
            else:
                generators.append(_rot_z(2 * n) @ SIGMA_H)  # D2d has S4
        elif suffix in ("i", "j"):
                generators.append(INVERSION)


def _close_group(generators, tol=1e-5):
    group = [np.eye(3)]
    queue = [np.eye(3)] + [np.asarray(g, dtype=float) for g in generators]

    while queue:
        op = queue.pop(0)
        # Add if distinct within floating-point tolerance
        if not any(np.allclose(op, g, atol=tol) for g in group):
            group.append(op)
            for g in list(group):
                queue.append(np.round(op @ g, 7))
                queue.append(np.round(g @ op, 7))

    return group


def _get_sym_info(name):
    key = name.strip().lower().replace(" ", "")
    if key in HM_TO_SCHOENFLIES:
        key = HM_TO_SCHOENFLIES[key]
    else:
        key = key.replace("-", "")
        key = ALIASES.get(key, key)
    match = re.match(r'^([CDTSO]|Ci|Cs)(\d+)?([hvdij]?)$', key, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized symmetry identifier: '{name}'")
    family, n_str, suffix = match.groups()
    family = family.capitalize()
    n = int(n_str) if n_str else 1
    suffix = suffix.lower() if suffix else ""
    return family, n, suffix


# Hermann-Mauguin (International) to canonical Schoenflies
HM_TO_SCHOENFLIES = {
    # Triclinic & Monoclinic
    "1": "C1", "-1": "Ci", "2": "C2", "m": "Cs", "2/m": "C2h",
    # Orthorhombic
    "222": "D2", "mm2": "C2v", "mmm": "D2h",
    # Tetragonal (e.g., Nd-LSCO)
    "4": "C4", "-4": "S4", "4/m": "C4h", "422": "D4",
    "4mm": "C4v", "-42m": "D2d", "-4m2": "D2d", "4/mmm": "D4h",
    # Trigonal & Hexagonal (e.g., PdCoO2, PdCrO2)
    "3": "C3", "-3": "C3i", "32": "D3", "3m": "C3v", "-3m": "D3d",
    "6": "C6", "-6": "C3h", "6/m": "C6h", "622": "D6",
    "6mm": "C6v", "-6m2": "D3h", "-62m": "D3h", "6/mmm": "D6h",
    # Cubic
    "23": "T", "m-3": "Th", "432": "O", "-43m": "Td", "m-3m": "Oh"
}


def _rot_z(n):
    """Proper rotation by 2*pi / n around the z-axis."""
    theta = 2.0 * np.pi / n
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


# Standard axis reflections, 2-fold axes, and inversions
# 180° rotation around x-axis
C2_X = np.diag([1.0, -1.0, -1.0])
# Horizontal mirror across xy-plane (kz -> -kz)
SIGMA_H = np.diag([1.0, 1.0, -1.0])
# Vertical mirror across xz-plane (ky -> -ky)
SIGMA_V = np.diag([1.0, -1.0, 1.0])
# Inversion i (k -> -k)
INVERSION = -np.eye(3)
# Diagonal mirror across kx = ky
SIGMA_D = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])
# 3-fold rotation along cubic diagonal [1, 1, 1]
ROT_C3_111 = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])
