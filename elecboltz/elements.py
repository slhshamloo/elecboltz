import numpy as np
from .fermisurface import IsoEnergySurface
from .bandstructure import BandStructure
from .kernel import ScatteringKernel, CustomKernel, SumKernel
from .integrate import quad_points, quad_weights

import numpy as np
import scipy.sparse

from typing import Callable
from collections.abc import Sequence
from numbers import Real

from scipy.constants import e, hbar, angstrom
THz = 1e12


class FiniteElements:
    """Structure for handling the discretized finite elements
    of a single iso-energy surface.
    
    Parameters
    ----------
    band
        The band structure to be discretized into finite elements.
    energy
        The energy of the iso-energy surface to be discretized into
        finite elements.
    field
        The applied field. If not provided, the field will be assumed
        to be zero.
    frequency
        The frequency of the applied field. If not provided, the
        frequency will be assumed to be zero.
    scattering_kernel
        The scattering kernel to be used for the discretization of
        the scattering rate. If not provided, the scattering rate will
        be calculated from the scattering_rate parameter.
    scattering_rate
        The scattering rate to be used for the discretization of the
        scattering rate. If not provided, the scattering rate will be
        calculated from the scattering_kernel parameter.
    quadrature_order
        The order of the quadrature to be used for the discretization
        of the scattering kernel. Higher order quadratures will give
        more accurate results, but will be slower to compute.
    correct_curvature
        Whether to correct for the effect of curvature on the jacobians.
        If set to true, the Jacobians will be calculated from the mesh
        that has the face centroids touching the true iso-energy
        surface. This is closer to the true Jacobian, but it breaks
        mathematical convergence guarantees.
    """
    def __init__(self, band: BandStructure, energy: Real,
                 frequency: Real = 0, field:Sequence[Real] = (0, 0, 0),
                 scattering_rate: Real | None = None,
                 scattering_kernel: ScatteringKernel | None = None,
                 quadrature_order: int = 2, correct_curvature: bool = True):
        self.band = band
        self.energy = energy
        self.surface = band.surfaces[energy]
        self.frequency = frequency
        self.field = np.array(field)
        self.temperature = 0.0 # temporary placeholder
        self.scattering_rate = scattering_rate
        self.scattering_kernel = scattering_kernel
        self.quadrature_order = quadrature_order
        self.correct_curvature = correct_curvature

        self.is_scattering_saved = False
        self.derivative_term = None
        self.out_scattering = None
        self.differential_operator = None
        self.fem_to_kernel = None
        self.scattering_invlen = None
        self.in_scattering_matrix = None
        self.build_band_elements()

    @property
    def kpoints(self):
        return self.surface.kpoints

    @property
    def kfaces(self):
        return self.surface.kfaces

    def build_band_elements(self):
        """
        Build the arrays corresponding to the discretization of the
        band structure.
        """
        self.velocities = np.column_stack(self.band.velocity_func(
            self.surface.kpoints[:, 0], self.surface.kpoints[:, 1],
            self.surface.kpoints[:, 2]))
        self.vmags = np.linalg.norm(self.velocities, axis=1)
        vhats = self.velocities / self.vmags[:, None]

        triangle_points = self.surface.kpoints[self.surface.kfaces] / angstrom
        if self.correct_curvature:
            triangle_points = self.band._curvature_correct_points(
                triangle_points, vhats[self.band.kfaces])
        if self.scattering_kernel is not None:
            self._quadrature_points = \
                quad_points[self.quadrature_order] @ triangle_points
    
        self._calculate_jacobian_sums(triangle_points)
        self._calculate_derivative_sums(triangle_points)
        self._calculate_velocity_projections(vhats)

    def erase_cache(self):
        self.fem_to_kernel = None
        self.scattering_invlen = None
        self.in_scattering_matrix = None
        self.out_scattering = None
        self.is_scattering_saved = False

    def _calculate_jacobian_sums(self, triangle_points):
            """Calculate the Jacobian sums for each point and point pair."""
            self.jacobians = np.linalg.norm(
                    np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                             triangle_points[:, 2] - triangle_points[:, 0]),
                    axis=-1)
            # build diagonal ordered matrices of the jacobian sums
            n = len(self.surface.kpoints)
            i_idx = self.surface.kfaces[:, 0]
            j_idx = self.surface.kfaces[:, 1]
            k_idx = self.surface.kfaces[:, 2]
            rows = np.concatenate((i_idx, j_idx, k_idx, i_idx, i_idx,
                                   j_idx, j_idx, k_idx, k_idx))
            cols = np.concatenate((i_idx, j_idx, k_idx, j_idx, k_idx,
                                   i_idx, k_idx, i_idx, j_idx))
            self.jacobian_sums = scipy.sparse.csc_array(
                (np.tile(self.jacobians, 9), (rows, cols)), shape=(n, n))
            self.overlap_matrix = self.jacobian_sums / 24
            self.overlap_matrix.setdiag(2 * self.overlap_matrix.diagonal())
    
    def _calculate_derivative_sums(self, triangle_points):
        """
        Calculate the field-independent part of the derivative term.
        """
        derivative_components = \
            triangle_points - np.roll(triangle_points, -2, axis=1)
        i_idx = self.surface.kfaces
        j_idx = np.roll(self.surface.kfaces, -1, axis=1)
        k_idx = np.roll(self.surface.kfaces, -2, axis=1)
        rows = np.concatenate((i_idx.flat, k_idx.flat))
        cols = np.tile(j_idx.flat, 2)
        self.derivative_sums = [
            scipy.sparse.csc_array((
            np.tile(component.flat, 2), (rows, cols))) for component in
            derivative_components.transpose(2, 0, 1)]
        if self.band.periodic:
            self.derivative_sums = [
                (self.surface.periodic_projector @ derivative
                    @ self.surface.periodic_projector.T).tocsc()
                for derivative in self.derivative_sums]

    def _calculate_velocity_projections(self, vhats):
        self.vhat_projections = self.overlap_matrix.tocsr() @ vhats
        if self.band.periodic:
            self.vhat_projections = \
                self.surface.periodic_projector @ self.vhat_projections

    def _build_differential_operator(self):
        """
        Build the differential operator from the elements of the
        band structure and the conductivity information.
        """
        if not self.is_scattering_saved:
            self._discretize_scattering()
            self._build_out_scattering_matrix()
            self.is_scattering_saved = True
        if self.derivative_term is None:
            self.derivative_term = sum(
                Bi / 6 * Di for Bi, Di
                in zip(self.field, self.derivative_sums))
        self.differential_operator = \
            self.out_scattering - e/hbar*self.derivative_term

    def _discretize_scattering(self):
        """
        Discretize the scattering rate and the scattering kernel
        for each element.
        """
        if self.scattering_kernel is None:
            self._discretize_independent_out_scattering()
            return

        if isinstance(self.scattering_kernel, SumKernel):
            self.fem_to_kernel = np.empty((0, len(self.surface.kpoints)))
            fem_basis_scattering_rate = np.zeros(len(self.surface.kpoints))
            coeffs = np.empty((0, 0))
            if self.scattering_kernel.custom_kernel is not None:
                fem_to_kernel, scattering_rate = \
                    self._discretize_custom_kernel(
                        self.scattering_kernel.custom_kernel)
                coeffs = scipy.linalg.block_diag(
                    coeffs, self.scattering_kernel.custom_kernel.coeffs)
                self.fem_to_kernel = np.vstack(
                    (self.fem_to_kernel, fem_to_kernel))
                fem_basis_scattering_rate += scattering_rate
            if len(self.scattering_kernel.explicit_kernels) > 0:
                fem_to_kernel, scattering_rate = \
                    self._discretize_explicit_kernel(self.scattering_kernel)
                coeffs = scipy.linalg.block_diag(
                    coeffs, self.scattering_kernel.coeffs)
                self.fem_to_kernel = np.vstack(
                    (self.fem_to_kernel, fem_to_kernel))
                fem_basis_scattering_rate += scattering_rate
        elif isinstance(self.scattering_kernel, CustomKernel):
            self.fem_to_kernel, fem_basis_scattering_rate = \
                self._discretize_custom_kernel(self.scattering_kernel)
            coeffs = self.scattering_kernel.coeffs
        else:
            self.fem_to_kernel, fem_basis_scattering_rate = \
                self._discretize_explicit_kernel(self.scattering_kernel)
            coeffs = self.scattering_kernel.coeffs
        self.in_scattering_matrix = -angstrom**2 * THz * coeffs
        self._calculate_scattering_invlen(fem_basis_scattering_rate)
    
    def _discretize_custom_kernel(self, kernel):
        kernel.decompose(self.band)
        fem_to_kernel = (self.overlap_matrix @ kernel.projector).T
        area_sums = self.jacobian_sums.diagonal() / 6
        fem_basis_scattering_rate = angstrom**2 * (
            (area_sums[None, :] @ kernel.projector)
            @ kernel.coeffs @ kernel.projector.T).flatten()
        return fem_to_kernel, fem_basis_scattering_rate

    def _discretize_explicit_kernel(self, kernel):
        fem_to_kernel = np.zeros(
            (kernel.coeffs.shape[0], len(self.surface.kpoints)),
            dtype=kernel.coeffs.dtype)
        sym_basis_scattering_rate = self._calculate_kernel_quadratures(
            kernel, fem_to_kernel)
        # (U^(-1))^i_a = sum_j (M^(-1))^ij U^dagger_ja
        transformed_scatrate = \
            fem_to_kernel.conj().T @ sym_basis_scattering_rate
        overlap_matrix_factor = scipy.sparse.linalg.splu(
            self.overlap_matrix)
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
                    kx=self.surface.kpoints[:, 0],
                    ky=self.surface.kpoints[:, 1],
                    kz=self.surface.kpoints[:, 2],
                    vx=self.velocities[:, 0],
                    vy=self.velocities[:, 1],
                    vz=self.velocities[:, 2],
                    E=self.energy,
                    T=self.temperature,
                    energy=self.energy,
                    temperature=self.temperature)
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
                np.add.at(fem_to_kernel[a], self.surface.kfaces[:, vertex],
                            self.jacobians / 2 * integrals.flatten())
        return self._integrate_kernel_quadratures(
            kernel, scattering_quadratures)
    
    def _integrate_kernel_quadratures(self, kernel, scattering_quadratures):
        sym_basis_scattering_rate = np.zeros(
            kernel.coeffs.shape[0], dtype=kernel.coeffs.dtype)
        weights = quad_weights[self.quadrature_order]
        for a in range(kernel.coeffs.shape[0]):
            sym_basis_integral = np.sum(
                self.jacobians / 2 # triangle areas
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
            self.scattering_invlen = THz * scattering / self.vmags
        else:
            self.scattering_invlen = \
                THz * (scattering - 2j*np.pi*self.frequency) / self.vmags

    def _build_out_scattering_matrix(self):
        """Calculate the out-scattering matrix (Gamma)"""
        self.out_scattering = (
            # alpha_ij * gamma^i
            (self.jacobian_sums * self.scattering_invlen[:, None]).tocsc()
            # alpha_ij * gamma^j
            + (self.jacobian_sums * self.scattering_invlen[None, :]).tocsc()
            # sum_k alpha_ik * gamma^k
            + scipy.sparse.diags_array(
                self.jacobian_sums @ self.scattering_invlen, format='csc')
            ) / 60
        # alpha(i,j,k) * gamma^k / 120
        i_idx = self.surface.kfaces[:, 0]
        j_idx = self.surface.kfaces[:, 1]
        k_idx = self.surface.kfaces[:, 2]
        n = len(self.surface.kpoints)
        rows = np.concatenate((i_idx, i_idx, j_idx, j_idx, k_idx, k_idx))
        cols = np.concatenate((j_idx, k_idx, i_idx, k_idx, i_idx, j_idx))
        data = np.concatenate((
            self.jacobians * self.scattering_invlen[k_idx],
            self.jacobians * self.scattering_invlen[j_idx],
            self.jacobians * self.scattering_invlen[k_idx],
            self.jacobians * self.scattering_invlen[i_idx],
            self.jacobians * self.scattering_invlen[j_idx],
            self.jacobians * self.scattering_invlen[i_idx])) / 120
        self.out_scattering += scipy.sparse.csc_array(
            (data, (rows, cols)), shape=(n, n))
        if self.band.periodic:
            self.out_scattering = (
                self.surface.periodic_projector @ self.out_scattering
                @ self.surface.periodic_projector.T).tocsc()
