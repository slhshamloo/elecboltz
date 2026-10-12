from .bandstructure import BandStructure
from .kernel import ScatteringKernel, CustomKernel, SumKernel
from .integrate import quad_points, quad_weights

import numpy as np
import scipy.sparse

from typing import Callable
from collections.abc import Sequence
from numbers import Real
from  numpy.typing import NDArray

from scipy.constants import e, hbar, angstrom
THz = 1e12


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
        self._velocities = None
        self._vmags = None
        self._vhat_projections = None
        self._quadrature_points = None
        self._jacobians = None
        self._jacobian_sums = None
        self._overlap_matrix = None
        self._derivatives = None
        self._fem_to_kernel = None
        self._scattering_invlen = None
        self._in_scattering_matrix = None
        self._out_scattering = None
        self._derivative_term = None
        self._differential_operator = None
        self._are_elements_saved = False
        self._is_scattering_saved = False

    def __setattr__(self, name, value):
        if name == 'band':
            self.erase_memory()
            super().__setattr__(name, value)
        if name in ['frequency', 'scattering_rate', 'scattering_kernel']:
            self.erase_memory(elements=False, scattering=True,
                              derivative=False)
            if name == 'scattering_rate' and isinstance(value, Sequence):
                if len(self.band.kpoints) != len(value):
                    raise ValueError(
                        "The scattering_rate must have the same length as "
                        "the number of kpoints in the band structure.")
                value = np.array(value)
            super().__setattr__(name, value)
        if name in ['field', 'Bamp', 'Btheta', 'Bphi']:
            self.erase_memory(elements=False, scattering=False,
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
                if self._derivative_term is not None:
                    self._derivative_term *= \
                        new_magnitude / self._field_magnitude
                if self._differential_operator is not None:
                    self._differential_operator = \
                        self._out_scattering - e/hbar*self._derivative_term
            else:
                self.erase_memory(elements=False, scattering=False,
                                  derivative=True)
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
        if not self._are_elements_saved:
            self._build_elements()
        if self._differential_operator is None:
            self._build_differential_operator()
        
        i, j = self._get_calculation_indices(i, j)
        # (A^{-1})^{ij} (v_b)_j
        self._linear_solution = self._solve(j)
        if len(self._linear_solution.shape) == 1:
            self._linear_solution = self._linear_solution[:, None]
        # (v_a)_i (A^{-1} v_b)^i
        sigma_result = self._vhat_projections[:, i].T @ self._linear_solution
        sigma_result *= e**2 / (4 * np.pi**3 * hbar) / self.band.bz_ratio
        if self.frequency == 0.0:
            sigma_result = sigma_result.real

        for idx_row, row in enumerate(i):
            for idx_col, col in enumerate(j):
                self.sigma[row, col] = sigma_result[idx_row, idx_col]
        return sigma_result

    def erase_memory(self, elements: bool = True, scattering: bool = True,
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
            self._velocities = None
            self._vmags = None
            self._quadrature_points = None
            self._jacobians = None
            self._jacobian_sums = None
            self._overlap_matrix = None
            self._derivatives = None
            self._vhat_projections = None
            self._are_elements_saved = False
        if scattering:
            self._fem_to_kernel = None
            self._scattering_invlen = None
            self._in_scattering_matrix = None
            self._out_scattering = None
            self._is_scattering_saved = False
        if derivative:
            self._derivative_term = None
        self._differential_operator = None
        self._factorization = None
        self._linear_solution = None
    
    def _solve(self, j):
        if self.scattering_kernel is None:
            self._factorization = scipy.sparse.linalg.splu(
                self._differential_operator)
            return self._factorization.solve(self._vhat_projections[:, j])
        else:
            # S_ij = C_ij / |v|_i, therefore we scale one of the U or V
            # matrices by the velocity magnitude to include the 1/|v|
            # factor in the transformed, full matrix
            U = self._fem_to_kernel.conj().T / self._vmags[:, None]
            V = self._fem_to_kernel
            # A_periodic = P A_0 P^dagger + P U C V P^dagger
            if self.band.periodic:
                U = self.band.periodic_projector @ U
                V = V @ self.band.periodic_projector.T
            self._factorization = scipy.sparse.linalg.splu(
                self._differential_operator)
            # A = A_0 + U^dagger S U
            return solve_sparse_plus_lowrank(
                self._differential_operator, self._in_scattering_matrix,
                U, V, self._vhat_projections[:, j],
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

    def _build_elements(self):
        """
        Build the arrays corresponding to the discretization of the
        band structure.
        """
        self._velocities = np.column_stack(self.band.velocity_func(
            self.band.kpoints[:, 0], self.band.kpoints[:, 1],
            self.band.kpoints[:, 2]))
        self._vmags = np.linalg.norm(self._velocities, axis=1)
        vhats = self._velocities / self._vmags[:, None]

        triangle_points = self.band.kpoints[self.band.kfaces] / angstrom
        if self.correct_curvature:
            triangle_points = self.band._curvature_correct_points(
                triangle_points, vhats[self.band.kfaces])
        if self.scattering_kernel is not None:
            self._quadrature_points = \
                quad_points[self.quadrature_order] @ triangle_points
    
        self._calculate_jacobian_sums(triangle_points)
        self._calculate_derivative_sums(triangle_points)
        self._calculate_velocity_projections(vhats)
        self._are_elements_saved = True

    def _calculate_jacobian_sums(self, triangle_points):
        """Calculate the Jacobian sums for each point and point pair."""
        self._jacobians = np.linalg.norm(
                np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                         triangle_points[:, 2] - triangle_points[:, 0]),
                axis=-1)
        # build diagonal ordered matrices of the jacobian sums
        n = len(self.band.kpoints)
        i_idx = self.band.kfaces[:, 0]
        j_idx = self.band.kfaces[:, 1]
        k_idx = self.band.kfaces[:, 2]
        rows = np.concatenate((i_idx, j_idx, k_idx, i_idx, i_idx,
                               j_idx, j_idx, k_idx, k_idx))
        cols = np.concatenate((i_idx, j_idx, k_idx, j_idx, k_idx,
                               i_idx, k_idx, i_idx, j_idx))
        self._jacobian_sums = scipy.sparse.csc_array(
            (np.tile(self._jacobians, 9), (rows, cols)), shape=(n, n))
        self._overlap_matrix = self._jacobian_sums / 24
        self._overlap_matrix.setdiag(2 * self._overlap_matrix.diagonal())

    def _calculate_derivative_sums(self, triangle_points):
        """
        Calculate the field-independent part of the derivative term.
        """
        derivative_components = (
            triangle_points - np.roll(triangle_points, -2, axis=1))
        i_idx = self.band.kfaces
        j_idx = np.roll(self.band.kfaces, -1, axis=1)
        k_idx = np.roll(self.band.kfaces, -2, axis=1)
        rows = np.concatenate((i_idx.flat, k_idx.flat))
        cols = np.tile(j_idx.flat, 2)
        self._derivatives = [
            scipy.sparse.csc_array((
            np.tile(component.flat, 2), (rows, cols))) for component in
            derivative_components.transpose(2, 0, 1)]
        if self.band.periodic:
            self._derivatives = [
                (self.band.periodic_projector @ derivative
                 @ self.band.periodic_projector.T).tocsc()
                for derivative in self._derivatives]

    def _calculate_velocity_projections(self, vhats):
        self._vhat_projections = self._overlap_matrix.tocsr() @ vhats
        if self.band.periodic:
            self._vhat_projections = \
                self.band.periodic_projector @ self._vhat_projections

    def _build_differential_operator(self):
        """
        Build the differential operator from the elements of the
        band structure and the conductivity information.
        """
        if not self._is_scattering_saved:
            self._discretize_scattering()
            self._build_out_scattering_matrix()
            self._is_scattering_saved = True
        if self._derivative_term is None:
            self._derivative_term = sum(
                Bi / 6 * Di for Bi, Di in zip(self.field, self._derivatives))
        self._differential_operator = \
            self._out_scattering - e/hbar*self._derivative_term

    def _discretize_scattering(self):
        """
        Discretize the scattering rate and the scattering kernel
        for each element.
        """
        if self.scattering_kernel is None:
            self._discretize_independent_out_scattering()
            return

        if isinstance(self.scattering_kernel, SumKernel):
            self._fem_to_kernel = np.empty((0, len(self.band.kpoints)))
            fem_basis_scattering_rate = np.zeros(len(self.band.kpoints))
            coeffs = np.empty((0, 0))
            if self.scattering_kernel.custom_kernel is not None:
                fem_to_kernel, scattering_rate = \
                    self._discretize_custom_kernel(
                        self.scattering_kernel.custom_kernel)
                coeffs = scipy.linalg.block_diag(
                    coeffs, self.scattering_kernel.custom_kernel.coeffs)
                self._fem_to_kernel = np.vstack(
                    (self._fem_to_kernel, fem_to_kernel))
                fem_basis_scattering_rate += scattering_rate
            if len(self.scattering_kernel.explicit_kernels) > 0:
                fem_to_kernel, scattering_rate = \
                    self._discretize_explicit_kernel(self.scattering_kernel)
                coeffs = scipy.linalg.block_diag(
                    coeffs, self.scattering_kernel.coeffs)
                self._fem_to_kernel = np.vstack(
                    (self._fem_to_kernel, fem_to_kernel))
                fem_basis_scattering_rate += scattering_rate
        elif isinstance(self.scattering_kernel, CustomKernel):
            self._fem_to_kernel, fem_basis_scattering_rate = \
                self._discretize_custom_kernel(self.scattering_kernel)
            coeffs = self.scattering_kernel.coeffs
        else:
            self._fem_to_kernel, fem_basis_scattering_rate = \
                self._discretize_explicit_kernel(self.scattering_kernel)
            coeffs = self.scattering_kernel.coeffs
        self._in_scattering_matrix = -angstrom**2 * THz * coeffs
        self._calculate_scattering_invlen(fem_basis_scattering_rate)
    
    def _discretize_custom_kernel(self, kernel):
        kernel.decompose(self.band)
        fem_to_kernel = (self._overlap_matrix @ kernel.projector).T
        area_sums = self._jacobian_sums.diagonal() / 6
        fem_basis_scattering_rate = angstrom**2 * (
            (area_sums[None, :] @ kernel.projector)
            @ kernel.coeffs @ kernel.projector.T).flatten()
        return fem_to_kernel, fem_basis_scattering_rate

    def _discretize_explicit_kernel(self, kernel):
        fem_to_kernel = np.zeros(
            (kernel.coeffs.shape[0], len(self.band.kpoints)),
            dtype=kernel.coeffs.dtype)
        sym_basis_scattering_rate = self._calculate_kernel_quadratures(
            kernel, fem_to_kernel)
        # (U^(-1))^i_a = sum_j (M^(-1))^ij U^dagger_ja
        transformed_scatrate = \
            fem_to_kernel.conj().T @ sym_basis_scattering_rate
        overlap_matrix_factor = scipy.sparse.linalg.splu(
            self._overlap_matrix)
        fem_basis_scattering_rate = \
            overlap_matrix_factor.solve(transformed_scatrate.real)
        if transformed_scatrate.dtype == complex:
            fem_basis_scattering_rate += 1j * overlap_matrix_factor.solve(
                transformed_scatrate.imag)
        return fem_to_kernel, fem_basis_scattering_rate

    def _discretize_independent_out_scattering(self):
        if self.scattering_rate is None:
            raise ValueError(
                "Either scattering_rate or scattering_kernel must be set.")
        else:
            if isinstance(self.scattering_rate, Callable):
                scattering = self.scattering_rate(
                    kx=self.band.kpoints[:, 0], ky=self.band.kpoints[:, 1],
                    kz=self.band.kpoints[:, 2], vx=self._velocities[:, 0],
                    vy=self._velocities[:, 1], vz=self._velocities[:, 2])
            else:
                scattering = self.scattering_rate
        self._calculate_scattering_invlen(scattering)
    
    def _calculate_kernel_quadratures(self, kernel, fem_to_kernel):
        scattering_quadratures = np.empty(
            (kernel.coeffs.shape[0],
             *self._quadrature_points.shape[:-1]))
        weights = quad_weights[self.quadrature_order]
        for a in range(kernel.coeffs.shape[0]):
            scattering_quadratures[a] = kernel.eval_basis(
                a, angstrom * self._quadrature_points[:, :, 0],
                angstrom * self._quadrature_points[:, :, 1],
                angstrom * self._quadrature_points[:, :, 2]).conj()
            for vertex in range(3):
                fem_basis = quad_points[self.quadrature_order][:, vertex]
                integrals = scattering_quadratures[a] @ (weights * fem_basis)
                np.add.at(fem_to_kernel[a], self.band.kfaces[:, vertex],
                          self._jacobians / 2 * integrals.flatten())
        return self._integrate_kernel_quadratures(kernel, scattering_quadratures)
    
    def _integrate_kernel_quadratures(self, kernel, scattering_quadratures):
        sym_basis_scattering_rate = np.zeros(
            kernel.coeffs.shape[0], dtype=kernel.coeffs.dtype)
        weights = quad_weights[self.quadrature_order]
        for a in range(kernel.coeffs.shape[0]):
            sym_basis_integral = np.sum(
                self._jacobians / 2 # triangle areas
                * (scattering_quadratures[a] @ weights[:, None]).flatten())
            for b in range(kernel.coeffs.shape[1]):
                # 1/tau_b = sum_a C_ab^* int dk psi_a(k)
                sym_basis_scattering_rate[b] += angstrom**2 * (
                    np.conj(kernel.coeffs[a, b]) * sym_basis_integral)
        return sym_basis_scattering_rate

    def _calculate_scattering_invlen(self, scattering):
        # scattering_invlen is the inverse scattering length gamma
        # separate the optical conductivity case to avoid making
        # the number complex when it is not needed
        if self.frequency == 0.0:
            self._scattering_invlen = THz * scattering / self._vmags
        else:
            self._scattering_invlen = \
                THz * (scattering - 2j*np.pi*self.frequency) / self._vmags

    def _build_out_scattering_matrix(self):
        """Calculate the out-scattering matrix (Gamma)"""
        self._out_scattering = (
            # alpha_ij * gamma^i
            (self._jacobian_sums * self._scattering_invlen[:, None]).tocsc()
            # alpha_ij * gamma^j
            + (self._jacobian_sums * self._scattering_invlen[None, :]).tocsc()
            # sum_k alpha_ik * gamma^k
            + scipy.sparse.diags_array(
                self._jacobian_sums @ self._scattering_invlen, format='csc')
            ) / 60
        # alpha(i,j,k) * gamma^k / 120
        i_idx = self.band.kfaces[:, 0]
        j_idx = self.band.kfaces[:, 1]
        k_idx = self.band.kfaces[:, 2]
        n = len(self.band.kpoints)
        rows = np.concatenate((i_idx, i_idx, j_idx, j_idx, k_idx, k_idx))
        cols = np.concatenate((j_idx, k_idx, i_idx, k_idx, i_idx, j_idx))
        data = np.concatenate((
            self._jacobians * self._scattering_invlen[k_idx],
            self._jacobians * self._scattering_invlen[j_idx],
            self._jacobians * self._scattering_invlen[k_idx],
            self._jacobians * self._scattering_invlen[i_idx],
            self._jacobians * self._scattering_invlen[j_idx],
            self._jacobians * self._scattering_invlen[i_idx])) / 120
        self._out_scattering += scipy.sparse.csc_array(
            (data, (rows, cols)), shape=(n, n))
        if self.band.periodic:
            self._out_scattering = (
                self.band.periodic_projector @ self._out_scattering
                @ self.band.periodic_projector.T).tocsc()


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
    return A_inv_b - A_inv_U @ np.linalg.solve(I + CV@A_inv_U, CV@A_inv_b)