from .bandstructure import BandStructure
from .kernel import ScatteringKernel
from .elements import FiniteElements

import numpy as np
import scipy.sparse

from typing import Callable
from collections.abc import Sequence
from numbers import Real
from numpy.typing import NDArray

from scipy.constants import e, hbar


class Conductivity:
    """
    Calculates the conductivity of a material solving the Boltzmann
    transport equation using a finite element method (FEM).

    Parameters
    ----------
    band
        The class holding band structure information of the material.
    field
        The magnetic field in the x, y, and z directions in units of
        Tesla.
    scattering_rate
        The (out-)scattering rate, in units of THz. If expressed as
        a ``Callable`` it is a function of any of the parameters:
        (wavevectors) kx, ky, kz, in units of 1/angstrom,
        (velocities) vx, vy, vz, in units of m/s,
        temperature in units of K, and energy (difference with the
        Fermi level) in units of meV. The function signature should
        have matching names for these parameters, and also collect
        any extra keyword arguments in ``**kwargs`` to keep the
        function signature consistent. Note that with this type of
        function signature, the function doesn't need to explicitly
        specify parameters that is does not use, and the order of the
        parameters also doesn't matter. Instead of a ``Callable``, it
        can also be a constant over the full Fermi surface, or set to
        custom values over the different ``kpoints`` of the Fermi
        surface in the `band` object. If None, it will be calculated
        from the scattering kernel.
    scattering_kernel
        See the `elecboltz.kernel` module for more details on this
        object. If provided, ``scattering_rate`` will be ignored
        and calculated from the kernel.
    frequency
        The frequency of the applied field in units of THz.
    correct_curvature
        If True, correct for the curvature of the Fermi surface.
    solver
        The solver used to solve the linear system. Takes the (sparse)
        matrix as the first argument and the right-hand side as the
        second argument. When using a custom solver, keep in mind that
        the right-hand side might not be a vector. So, solvers that
        only work with vectors need to be adapted to solve each column
        of the right-hand side separately.
    quadrature_order
        The order of the quadrature used to integrate the scattering
        kernel. 2 should be sufficient for most cases.
    Bamp
        The amplitude of the magnetic field in units of Tesla. If
        provided, it will be used to set the field.
    Btheta
        The polar angle of the magnetic field in units of degrees. If
        provided, it will be used to set the field.
    Bphi
        The azimuthal angle of the magnetic field in units of degrees.
        If provided, it will be used to set the field.
    
    Attributes
    ----------
    sigma : NDArray[complexfloating]
        The conductivity tensor, which is a 3 by 3 matrix. Can be
        calculated using the ``solve`` method. Elements that are not
        calculated yet are set to zero.
    """
    def __init__(
            self, band: BandStructure, field: Sequence[float] = np.zeros(3),
            scattering_rate: Callable | Sequence | Real | None = None,
            scattering_kernel: ScatteringKernel | None = None,
            frequency: Real = 0.0, correct_curvature: bool = True,
            quadrature_order: int = 2, Bamp: Real | None = None,
            Btheta: Real | None = None, Bphi: Real | None = None, **kwargs):
        self.correct_curvature = correct_curvature
        self.quadrature_order = quadrature_order
        # avoid triggering setattr in the constructor
        super().__setattr__('band', band)
        super().__setattr__('scattering_rate', scattering_rate)
        super().__setattr__('scattering_kernel', scattering_kernel)
        super().__setattr__('frequency', frequency)
        self._field_direction = None
        self.set_field(field, Bamp, Btheta, Bphi)
        self.sigma = np.zeros((3, 3))
        self._finite_elements = {}

    def __setattr__(self, name, value):
        if name == 'band':
            self.erase_cache()
            super().__setattr__(name, value)
        if name in ['frequency', 'scattering_rate', 'scattering_kernel']:
            self.erase_cache(elements=False, scattering=True,
                             derivative=False)
            if name == 'scattering_rate' and isinstance(value, Sequence):
                if len(self.band.kpoints) != len(value):
                    raise ValueError(
                        "The scattering_rate must have the same length as "
                        "the number of kpoints in the band structure.")
                value = np.array(value)
            super().__setattr__(name, value)
        if name in ['field', 'Bamp', 'Btheta', 'Bphi']:
            self.erase_cache(elements=False, scattering=False,
                              derivative=True)
            super().__setattr__(name, value)
            if name == 'field':
                super().__setattr__('Bamp', None)
                super().__setattr__('Btheta', None)
                super().__setattr__('Bphi', None)
            self.set_field(
                field=self.field, Bamp=self.Bamp,
                Btheta=self.Btheta, Bphi=self.Bphi)
        else:
            super().__setattr__(name, value)

    def set_field(self, field: Sequence[Real] = np.zeros(3),
                  Bamp: Real | None = None, Btheta: Real | None = None,
                  Bphi: Real | None = None):
        """Set the magnetic field.

        Note that you can both provide the field in Cartesian
        coordinates (x, y, z) or in spherical coordinates
        (amplitude, polar angle, azimuthal angle). If both are provided,
        the spherical coordinates will be used to set the field.

        Parameters
        ----------
        field
            If you want to directly set the field in Cartesian
            coordinates, provide the field as a sequence of three values
            in units of Tesla.
        Bamp
            If you want to set the field in spherical coordinates,
            provide the amplitude of the field in units of Tesla.
        Btheta
            If you want to set the field in spherical coordinates,
            provide the polar angle of the field in degrees.
        Bphi
            If you want to set the field in spherical coordinates,
            provide the azimuthal angle of the field in degrees.
        """
        if Bamp is not None:
            Btheta = Btheta or 0.0
            Bphi = Bphi or 0.0
            field = [
                Bamp * np.sin(np.deg2rad(Btheta)) * np.cos(np.deg2rad(Bphi)),
                Bamp * np.sin(np.deg2rad(Btheta)) * np.sin(np.deg2rad(Bphi)),
                Bamp * np.cos(np.deg2rad(Btheta))]
        field = np.array(field)
        new_magnitude = np.linalg.norm(field)
        if new_magnitude != 0:
            new_direction = field / new_magnitude
        else:
            new_direction = np.zeros(3)
        if self._field_direction is not None:
            if np.all(self._field_direction == new_direction):
                for element in self._finite_elements.values():
                    if element.derivative_term is not None:
                        element.derivative_term *= \
                            new_magnitude / self._field_magnitude
                    if element.differential_operator is not None:
                        element.differential_operator = (
                            element.out_scattering
                            - e/hbar*element.derivative_term)
            else:
                self.erase_cache(
                    elements=False, scattering=False, derivative=True)
        self._field_magnitude = new_magnitude
        self._field_direction = new_direction
        if Bamp is None:
            Bamp = new_magnitude
            Btheta = np.rad2deg(np.arccos(new_direction[2]))
            Bphi = np.rad2deg(np.arctan2(new_direction[1], new_direction[0]))
        super().__setattr__('field', field)
        super().__setattr__('Bamp', Bamp)
        super().__setattr__('Btheta', Btheta)
        super().__setattr__('Bphi', Bphi)

    def calculate(self, i: Sequence[int] | int | None = None,
                  j: Sequence[int] | int | None = None
                  ) -> NDArray[np.complexfloating]:
        """Calculate the conductivity tensor.

        Parameters
        ----------
        i
            The index of the first component (row) of the conductivity
            tensor. If None (default), all components are calculated.
        j
            The index of the second component (column) of the
            conductivity tensor. If None (default), all components
            are calculated.
        Returns
        -------
        NDArray or float
            The conductivity tensor component(s) as an i by j matrix.
        """
        if 0.0 not in self._finite_elements:
            self._build_finite_elements(0.0)
        if self._finite_elements[0.0].differential_operator is None:
            self._finite_elements[0.0]._build_differential_operator()
        
        i, j = self._get_calculation_indices(i, j)
        # (A^{-1})^{ij} (v_b)_j
        self._linear_solution = self._solve(j)
        if len(self._linear_solution.shape) == 1:
            self._linear_solution = self._linear_solution[:, None]
        # (v_a)_i (A^{-1} v_b)^i
        sigma_result = (self._finite_elements[0.0].vhat_projections[:, i].T
                        @ self._linear_solution)
        sigma_result *= e**2 / (4 * np.pi**3 * hbar) / self.band.bz_ratio
        if self.frequency == 0.0:
            sigma_result = sigma_result.real

        for idx_row, row in enumerate(i):
            for idx_col, col in enumerate(j):
                self.sigma[row, col] = sigma_result[idx_row, idx_col]
        return sigma_result

    def erase_cache(self, elements: bool = True, scattering: bool = True,
                    derivative: bool = True):
        """Erase saved calculations to free memory.

        This class saves already calculated values for the FEM elements
        and matrices to avoid recalculating them every time a new
        element of the conductivity tensor is calculated or a new field
        is applied. This method is provided to erase those values when
        no new calculations are needed and the memory can be freed.

        Parameters
        ----------
        elements : bool, optional
            If True, erase the quantities for each element, like the
            lengths and velocities.
        scattering : bool, optional
            If True, erase the out-scattering and in-scattering terms.
        derivative : bool, optional
            If True, erase the derivative term.
        """
        if elements:
            self._finite_elements = {}
        if scattering:
            for element in self._finite_elements.values():
                element.erase_cache()
        if derivative:
            for element in self._finite_elements.values():
                element.derivative_term = None
                element.differential_operator = None
        self._factorization = None
        self._linear_solution = None
    
    def _solve(self, j):
        elements = self._finite_elements[0.0] # zero temperature placeholder
        if self.scattering_kernel is None:
            self._factorization = scipy.sparse.linalg.splu(
                elements.differential_operator)
            return self._factorization.solve(
                elements.vhat_projections[:, j])
        else:
            # S_ij = C_ij / |v|_i, therefore we scale one of the U or V
            # matrices by the velocity magnitude to include the 1/|v|
            # factor in the transformed, full matrix
            U = elements.fem_to_kernel.conj().T / elements.vmags[:, None]
            V = elements.fem_to_kernel
            # A_periodic = P A_0 P^dagger + P U C V P^dagger
            if self.band.periodic:
                U = self.band.periodic_projector @ U
                V = V @ self.band.periodic_projector.T
            self._factorization = scipy.sparse.linalg.splu(
                elements.differential_operator)
            # A = A_0 + U^dagger S U
            return solve_sparse_plus_lowrank(
                elements.differential_operator, elements.in_scattering_matrix,
                U, V, elements.vhat_projections[:, j],
                sparse_solver = lambda _, b: self._factorization.solve(b))

    def _get_calculation_indices(self, i, j):
        if i is None:
            i = range(3)
        elif isinstance(i, int):
            i = [i]
        if j is None:
            j = range(3)
        elif isinstance(j, int):
            j = [j]
        return i, j

    def _build_finite_elements(self, energy):
        self._finite_elements[energy] = FiniteElements(
            band=self.band,
            energy=energy,
            frequency=self.frequency,
            field=self.field,
            scattering_rate=self.scattering_rate,
            scattering_kernel=self.scattering_kernel,
            quadrature_order=self.quadrature_order,
            correct_curvature=self.correct_curvature
        )

def solve_sparse_plus_lowrank(
        A, C, U, V, b, sparse_solver=scipy.sparse.linalg.spsolve):
    """Solve the linear system ``(A + U @ C @ V) x = b`` for x,
    where A is a sparse matrix, and U @ C @ V is a low-rank matrix.

    Using the Sherman--Morrison--Woodbury formula, the solution can
    be calculated as:

    .. math::
        x = A^{-1}b - A^{-1}U (C^{-1}+VA^{-1}U)^{-1} VA^{-1}b
    
    So, only two sparse solves :math:`A^{-1} b` and :math:`A^{-1} U`
    and two small dense solves :math:`C^{-1}` and
    :math:`(C^{-1} + V A^{-1} U)^{-1}` are needed, which is much more
    efficient than solving the full dense system when the rank of the
    low-rank part is small.

    Note: Since the matrix C might be singular, the implementation uses
    the euivalent form of the Sherman--Morrison--Woodbury formula

    .. math::
        x = A^{-1}b - A^{-1}U (I+CVA^{-1}U)^{-1} CVA^{-1}b
    
    which only requires inverting the matrix :math:`I + CV A^{-1} U`.

    Parameters
    ----------
    A : scipy.sparse matrix
        The sparse matrix part of the operator, with shape (n, n).
    C : NDArray
        A small dense matrix of shape (r, r), where r is the rank of
        the low-rank part.
    U : NDArray
        A matrix of shape (n, r) representing the left singular vectors
        of the low-rank part.
    V : NDArray
        A matrix of shape (r, n) representing the right singular vectors
        of the low-rank part.
    b : NDArray
        The right-hand side vector or matrix, with shape (n,) or (n, m).
    sparse_solver : Callable, optional
        The solver used to solve the sparse linear systems. Takes the
        sparse matrix as the first argument and the right-hand side as
        the second argument. When using a custom solver, keep in mind
        that the right-hand side might not be a vector.

    Returns
    -------
    NDArray
        The solution x to the linear system, with the same shape as b.
    """
    A_inv_b = sparse_solver(A, b)
    A_inv_U = sparse_solver(A, U)
    I = np.eye(C.shape[0])
    CV = C @ V
    try:
        dense_inv = np.linalg.solve(I + CV @ A_inv_U, CV @ A_inv_b)
    except np.linalg.LinAlgError:
        # Solve the equivalent bordered system without forming the dense
        # n-by-n low-rank update U @ C @ V.
        # The bordered system is:
        # [ A  U ] [ x ] = [ b ]
        # [ -CV I ] [ y ]   [ 0 ]
        # which is equivalent to the original (A + U @ C @ V)x = b.
        rank = C.shape[0]
        bordered_operator = scipy.sparse.bmat([
            [A, scipy.sparse.csc_array(U)],
            [-scipy.sparse.csc_array(CV),
             scipy.sparse.eye(rank, format='csc')]],
            format='csc')
        b = np.asarray(b)
        zero_block = np.zeros((rank, *b.shape[1:]), dtype=b.dtype)
        bordered_rhs = np.concatenate((b, zero_block), axis=0)
        bordered_solution = scipy.sparse.linalg.spsolve(
            bordered_operator, bordered_rhs)
        return bordered_solution[:A.shape[0]]
    return A_inv_b - A_inv_U @ dense_inv