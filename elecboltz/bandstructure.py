from .fermisurface import IsoEnergySurface
from .symmetry import Symmetry
from .integrate import adaptive_octree_integrate

import numpy as np
import sympy
import scipy.sparse
import scipy.optimize
from skimage.measure import marching_cubes

from collections.abc import Sequence
from numbers import Real

from scipy.constants import hbar, eV, angstrom, m_e
# conversion from energy gradient units to m/s for velocity
velocity_units = 1e-3 * eV * angstrom / hbar


class BandStructure:
    """Contains bandstructure information for a given material.

    In addition to the dispersion relation and general parameters, this
    class also contains methods for discretizing the Fermi surface and
    calculating electronic properties.

    Parameters
    ----------
    dispersion
        The dispersion relation. Expresses the dispersion relation
        in terms of symbols in ``wavevector_names`` and additional
        parameters in ``band_params``. It must be parsable and
        differentiable by ``sympy``. Energy units are milli eV.
    chemical_potential
        The chemical potential in milli eV.
    unit_cell
        The dimensions of the unit cell in angstrom.
    band_params
        The parameters of the dispersion relation. Energy units are
        milli eV and distance units are angstrom.
    symmetry
        The point group symmetry of the material. Can be the lattice
        type (e.g., 'tetragonal', 'trigonal', 'hexagonal'), or a
        symmetry symbol: a Schoenflies symbol (e.g., 'D4h', 'D3d',
        'C4v') or a Hermann-Mauguin symbol (e.g., '4/mmm', '-3m').
    fixed_filling
        The fixed electronic filling fraction (``n``) of the material.
        If not None, this is used to set the chemical potential upon
        discretization of the Feri surface. The actual ``n`` value
        might be slightly different.
    domain_size
        The ratio of the reciprocal space domain sidelengths to simple
        cubic unit cell dimensions in reciprocal space. The product
        of the numbers in this collection must be equal to the number
        of atoms in the conventional unit cell specified by
        ``unit_cell``.
    periodic
        If bool, whether periodic boundary conditions are applied to all
        axes or not. If a single int, specifies which single axis is 
        periodic. If a collection, specifies which axes are periodic.
        If the collection is of integers, the integers specify the
        periodic axes, e.g. ``[0, 2]`` means periodic in `x` and `z`
        axes. If the collection is of booleans, it specifies whether
        each axis is periodic or not, e.g. ``[True, False, True]`` means
        periodic in `x` and `z` axes, but not in `y` axis.
    axis_names
        The names of the unit cell axes. Must be parsable by
        `sympy.symbols`.
    wavevector_names
        The names of the wavevector components. Must be parsable by
        `sympy.symbols`.
    resolution
        Controls the resolution of the grids used for discretizing the
        Fermi surface. If a collection of integers is provided, each
        element corresponds to the resolution along the respective
        axis. If a single integer is provided, it is used for all axes.
    filling_tuning_depth
        The depth of the adaptive octree integration used for calculating
        the filling fraction when tuning the chemical potential to match
        the fixed filling fraction.
    n_correct
        The number of correction steps for improving the accuracy of
        the discretization of the Fermi surface.
    sort_axis
        The axis along which to sort the points after triangulation.
        If None, do not sort the points.

    Attributes
    ----------
    surfaces : Mapping[float, FermiSurface]
        A mapping from energy levels to the corresponding discretized
        iso energy surfaces.
    n : float
        The electron filling fraction of the material. Only available
        after calling ``calculate_filling_fraction``.
    p : float
        The hole filling fraction of the material. Only available
        after calling ``calculate_filling_fraction``.
    m : float
        The effective mass of the charge carriers divided by the
        rest mass of the electron, m_e. Only available after calling
        ``calculate_mass``.
    """
    def __init__(
            self, dispersion: str, chemical_potential: Real,
            unit_cell: Sequence[Real], band_params: dict = {},
            symmetry: str | None = None, fixed_filling: Real | None = None,
            domain_size: Sequence[Real] = np.ones(3), bz_ratio: Real = 1.0,
            periodic: bool | int | Sequence[int | bool] = 2,
            axis_names: Sequence[str] | str = ['a', 'b', 'c'],
            wavevector_names: Sequence[str] | str = ['kx', 'ky', 'kz'],
            resolution: int | Sequence[int] = 31,
            filling_tuning_depth: int = 6, n_correct: int = 2,
            sort_axis: int | None = None, **kwargs):
        # avoid triggering the __setattr__ method for the first time
        super().__setattr__('dispersion', dispersion)
        self.band_params = band_params
        self.symmetry = symmetry
        if symmetry is not None:
            self.symmetry = Symmetry(self.symmetry)
        self.fixed_filling = fixed_filling
        self.chemical_potential = chemical_potential
        self.unit_cell = unit_cell
        self.domain_size = domain_size
        self.bz_ratio = bz_ratio
        self.periodic = periodic
        self.axis_names = axis_names
        self.wavevector_names = wavevector_names
        self.resolution = resolution
        self.filling_tuning_depth = filling_tuning_depth
        self.n_correct = n_correct
        self._parse_dispersion()
        self.surfaces = {}
        self.sort_axis = sort_axis
        self.n = None
        self.p = None
        self.m = None

    def __setattr__(self, name, value):
        if name == 'dispersion':
            self._parse_dispersion()
        if name == 'resolution':
            if isinstance(value, Sequence):
                value = np.array(value)
            else:
                value = np.array([value, value, value])
        if name in {'unit_cell', 'domain_size'}:
            value = np.array(value, dtype=float)
        if name == 'periodic':
            if isinstance(value, bool):
                if value:
                    value = [0, 1, 2]
                else:
                    value = []
            if isinstance(value, int):
                value = [value]
            elif isinstance(value, Sequence) and all(
                    isinstance(i, bool) for i in value):
                value = [i for i, v in enumerate(value) if v]
        super().__setattr__(name, value)
    
    def __getstate__(self):
        """Get the state of the object for pickling."""
        state = self.__dict__.copy()
        # remove the full energy and velocity functions
        state['_energy_func_full'] = None
        state['_velocity_funcs_full'] = None
        return state
    
    def __setstate__(self, state):
        """Set the state of the object after unpickling."""
        self.__dict__.update(state)
        # re-parse the dispersion relation to restore the full functions
        self._parse_dispersion()

    def discretize(self, energy: float = 0.0):
        """Discretize the iso-energy surface at the given energy level.

        First, the surface is triangulated using the marching cubes
        algorithm with ``resolution`` controlling the resolution of the
        grid. Next, to improve the accuracy of the isosurface,
        ``n_correct`` steps of  the Newton--Raphson root-finding method
        are applied to the output of marching cubes. Finally, after the
        surface construction, periodic boundary conditions are applied
        to "stitch" the open ends of the surface together.

        Parameters
        ----------
        energy
            The difference of the energy level of the discretized
            surface from the chemical potential. The default is 0.0,
            which corresponds to the Fermi surface.
        """
        surface = self._build_surface(energy)
        self.surfaces[energy] = surface
        if self.symmetry is not None:
            surface.symmetrize(self.symmetry)
        if energy == 0 and self.fixed_filling is not None:
            self.tune_chemical_potential()
        if self.sort_axis is not None:
            surface.sort_and_reindex(self.sort_axis)
        surface.apply_periodicity(self._gvec, self.periodic)
        return surface

    def calculate_filling_fraction(self, depth: int = 7) -> float:
        """Calculate the filling fraction n of the material.

        The filling fraction is calculated by integrating the volume
        in the reciprocal space having energy bellow the Fermi level
        (calculated by an adaptive octree integration method), then
        dividing by the volume of the unit cell in the reciprocal space.

        Parameters
        ----------
        depth
            The depth of the adaptive octree integration. Higher values
            result in more accurate integration, but take exponentially
            longer to compute.

        Returns
        -------
        float
            The filling fraction n of the material.
        """
        self._gvec = self.domain_size * np.pi / self.unit_cell
        # the extra factor of 2 is the spin degeneracy
        self.n =  2 * adaptive_octree_integrate(
            lambda kx, ky, kz: (self.energy_func(kx, ky, kz)
                                < self.chemical_potential),
            (-self._gvec[0], self._gvec[0], -self._gvec[1], self._gvec[1],
             -self._gvec[2], self._gvec[2]), depth=depth
            ) / 8 / np.prod(self._gvec) / self.bz_ratio
        self.p = 1 - self.n
        return self.n

    def calculate_electron_density(self, depth: int = 7) -> float:
        """Calculate the electron density n_e of the material.

        Note that the surface needs to be discretized before calling
        this method. First, the filling fraction is calculated (see
        ``calculate_filling_fraction``). The electron density is
        obtained by dividing the filling fraction by the volume of
        the unit cell in real space.

        Parameters
        ----------
        depth
            The depth of the adaptive octree integration in
            ``calculate_filling_fraction``.

        Returns
        -------
        float
            The electron density n_e of the material in SI units.
        """
        filling_fraction = self.calculate_filling_fraction(depth)
        # the volume of the unit cell in real space is scaled by
        # the inverse scaling of the unit cell in reciprocal space
        unit_cell_volume = (np.prod(self.unit_cell) * angstrom**3
                            / np.prod(self.domain_size))
        return filling_fraction / unit_cell_volume

    def calculate_mass(self):
        """Calculate the effective mass of the charge carries.

        Returns
        -------
        float
            The effective mass divided by the rest mass of
            the electron, m_e.
        """
        # Placeholder for actual calculation
        triangle_points = self.kpoints[self.kfaces] / angstrom
        vs = np.column_stack(self.velocity_func(
            self.kpoints[:, 0], self.kpoints[:, 1], self.kpoints[:, 2]))
        vhats = vs / np.linalg.norm(vs, axis=-1)[:, None]
        normals = vhats[self.kfaces]
        triangle_points = self._curvature_correct_points(
            triangle_points, normals)
        dks = np.linalg.norm(
            np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                     triangle_points[:, 2] - triangle_points[:, 0]),
            axis=-1)
        ks = np.mean(triangle_points, axis=1)
        vs = self.velocity_func(
            ks[:, 0] * angstrom, ks[:, 1] * angstrom, ks[:, 2] * angstrom)
        k_perp = np.sqrt(ks[:, 0]**2 + ks[:, 1]**2)
        v_perp = np.sqrt(vs[0]**2 + vs[1]**2)
        self.m = np.sum(hbar * k_perp / v_perp * dks) / np.sum(dks) / m_e
        return self.m
    
    def tune_chemical_potential(self) -> float:
        """Tune the ``chemical_potential`` to match the electron
        ``fixed_filling`` fraction.
        
        Returns
        -------
        float
            The tuned chemical potential in milli eV.
        """
        def filling_diff(mu):
            if 'mu' in self.band_params:
                self.band_params['mu'] = mu
            else:
                self.chemical_potential = mu
            return (self.calculate_filling_fraction(
                        depth=self.filling_tuning_depth)
                    - self.fixed_filling)
        energy_scale_guess = max(np.abs(self.energy_func(0, 0, 0)),
                                 np.abs(self.energy_func(*self._gvec)))
        bracket = [-10 * energy_scale_guess, 10 * energy_scale_guess]
        mu = scipy.optimize.brentq(filling_diff, *bracket)
        if 'mu' in self.band_params:
            self.band_params['mu'] = mu
        else:
            self.chemical_potential = mu
        return mu

    def energy_func(self, kx, ky, kz):
        """Calculate the energy at the given k-point.

        Parameters
        ----------
        kx, ky, kz : float
            The components of the wavevector in angstrom^-1.

        Returns
        -------
        object like kx, ky, kz
            The energy at the given k-point in milli eV.
        """
        return self._energy_func_full(
            kx, ky, kz, *self.unit_cell, **self.band_params)
    
    def velocity_func(self, kx, ky, kz):
        """Calculate the velocity at the given k-point.

        Parameters
        ----------
        kx, ky, kz : float
            The components of the wavevector in angstrom^-1.

        Returns
        -------
        list of 3 objects like kx, ky, kz
            The velocity vector at the given k-point in m/s.
        """
        return [vfunc(kx, ky, kz, *self.unit_cell, **self.band_params)
                for vfunc in self._velocity_funcs_full]

    # For convenient access to the Fermi surface points and faces
    @property
    def kpoints(self):
        return self.surfaces[0.0].kpoints if self.surfaces else None

    @property
    def kfaces(self):
        return self.surfaces[0.0].kfaces if self.surfaces else None

    @property
    def periodic_projector(self):
        return self.surfaces[0.0].periodic_projector if self.surfaces else None

    @property
    def symmetry_projector(self):
        return (self.surfaces[0.0].symmetry_projector
                if self.surfaces else None)

    def _parse_dispersion(self):
        """
        Parse the dispersion relation and extract the necessary
        information for further calculations.
        """
        ksymbols = sympy.symbols(self.wavevector_names)
        all_symbols = (ksymbols + sympy.symbols(self.axis_names)
                       + sympy.symbols(list(self.band_params.keys())))
        # symbolic expressions
        self._energy_sympy = sympy.sympify(self.dispersion)
        self._velocities_sympy = [
            sympy.diff(self._energy_sympy, k) * velocity_units
            for k in sympy.symbols(self.wavevector_names)]
        # replace zero velocities with zero arrays
        for i, v in enumerate(self._velocities_sympy):
            if v == 0:
                self._velocities_sympy[i] = f"numpy.zeros_like({ksymbols[i]})"
        # convert expressions into python functions
        self._energy_func_full = sympy.lambdify(
            all_symbols, self._energy_sympy)
        self._velocity_funcs_full = [
            sympy.lambdify(all_symbols, vexpr, 'numpy')
            for vexpr in self._velocities_sympy]
    
    def _build_surface(self, energy: Real = 0.0):
        self._gvec = self.domain_size * np.pi / self.unit_cell
        kpoints, kfaces, _, _ = marching_cubes(
            self.energy_func(*np.mgrid[
                -self._gvec[0]:self._gvec[0]:1j*self.resolution[0],
                -self._gvec[1]:self._gvec[1]:1j*self.resolution[1],
                -self._gvec[2]:self._gvec[2]:1j*self.resolution[2]]),
            level=self.chemical_potential + energy)
        kpoints *= (2*self._gvec / (self.resolution-1))[None, :]
        kpoints -= self._gvec[None, :]
        for _ in range(self.n_correct):
            kpoints = self._apply_newton_correction(kpoints, energy)
        return IsoEnergySurface(kpoints, kfaces)

    def _apply_newton_correction(self, points, energy: Real = 0.0):
        return _apply_newton_correction(
            points, self.energy_func,
            lambda kx, ky, kz: (np.array(self.velocity_func(kx, ky, kz))
                                / velocity_units),
            self.chemical_potential + energy)
    
    def _curvature_correct_points(self, points, normals, energy: Real = 0.0):
        kcenters = np.mean(points, axis=1) * angstrom
        kcenters_tangent = kcenters.copy()
        for _ in range(2):
            kcenters_tangent = self._apply_newton_correction(
                kcenters_tangent, energy)
        center_diff = (kcenters_tangent-kcenters) / angstrom
        projected_diff = np.linalg.norm(center_diff, axis=-1)

        # Turn off warnings for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # cosine = nhat.cdiff / |cdiff||nhat| and |nhat| = 1
            cosines = np.einsum('ijk,ik->ij', normals, center_diff
                                ) / projected_diff[:, None]
            # Handle division by zero
            np.nan_to_num(cosines, copy=False, nan=1.0)
        diff = projected_diff[:, None] / cosines
        return points + normals*diff[:, :, None]


def _apply_newton_correction(points, func, gradient, iso_value):
    """
    Apply one step of the Newton-Raphson method to correct the points
    on the isosurface.

    Parameters
    ----------
    points : NDArray
        The points to be corrected.
    iso_value : float
        The isosurface value.
    gradient : callable
        A function that computes the gradient of the scalar field at
        given points.

    Returns
    -------
    NDArray
        The corrected points.
    """
    residuals = func(points[:, 0], points[:, 1], points[:, 2]) - iso_value
    gradient_vectors = np.column_stack(gradient(
        points[:, 0], points[:, 1], points[:, 2]))
    gradient_norms = np.linalg.norm(gradient_vectors, axis=-1)
    return points - (residuals/gradient_norms**2)[:, None]*gradient_vectors
