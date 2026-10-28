from .bandstructure import BandStructure
from .conductivity import Conductivity
from .fermisurface import IsoEnergySurface
from .params import easy_params, _deep_update
from .scatrate import build_scattering_function

import numpy as np
import scipy
import json
from datetime import datetime
from time import time
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Callable
from collections.abc import Sequence, Collection, Mapping
from numbers import Real
from numpy.typing import NDArray
from scipy.constants import e, hbar, angstrom
THz = 1e12


class FittingRoutine:
    """This convenience class automatically sets up a fitting routine.

    Stores the necessary data and generates the functions used in
    This includes handling the residual computation, parameter update,
    logging the fitting progress, and saving the results to a file.
    
    Attributes
    ----------
    params : Mapping
        The current parameters dictionary.
    iteration : int
        The current iteration number.
    last_time : float
        The timestamp of the last fitting iteration.
    total_time : float
        The total time spent on the fitting routine.
    """
    def __init__(self, init_params: Mapping, param_keys: Sequence[str],
                 x_data: Mapping | Sequence[Mapping],
                 y_data: Mapping | Sequence[Mapping],
                 x_shift: Mapping | Sequence[Mapping] = None,
                 x_normalize: Mapping | Sequence[Mapping] = None,
                 y_shift: Mapping | Sequence[Mapping] = None,
                 y_normalize: Mapping | Sequence[Mapping] = None,
                 loss: Callable = lambda y_fit, y_data: np.mean(
                     np.abs(y_fit - y_data)),
                 postprocess: Callable = lambda x, y: y, save_path: str = None,
                 save_label: str = "fit", update_keys: Collection[str] = None,
                 multi_params: Collection[str] = None,
                 multi_params_labels: Collection[str] = None,
                 print_log: bool = True):
        self.init_params = init_params
        if multi_params:
            self._init_params_multi = deepcopy(init_params)
        self.param_keys = param_keys
        self.x_data_sets = _normalize_data_sets(x_data, "x_data")
        self.y_data_sets = _normalize_data_sets(y_data, "y_data")
        _validate_data_sets(self.x_data_sets, self.y_data_sets)
        self.x_data = self.x_data_sets[0]
        self.y_data = self.y_data_sets[0]
        self.x_shift = _normalize_optional_data_sets(
            x_shift, len(self.x_data_sets), "x_shift")
        self.x_normalize = _normalize_optional_data_sets(
            x_normalize, len(self.x_data_sets), "x_normalize")
        self.y_shift = _normalize_optional_data_sets(
            y_shift, len(self.x_data_sets), "y_shift")
        self.y_normalize = _normalize_optional_data_sets(
            y_normalize, len(self.x_data_sets), "y_normalize")
        if multi_params and (len(self.x_data_sets) > 1 or
                             len(self.y_data_sets) > 1):
            raise ValueError(
                "Combining dataset mappings with multi_params is not "
                "supported; use the existing nested mapping format.")
        self.loss = loss
        self.postprocess = postprocess
        self.save_path = save_path
        self.save_label = save_label
        self.update_keys = update_keys
        self.multi_params = multi_params
        self.multi_params_labels = multi_params_labels
        self.print_log = print_log
        self.iteration = 0
        self.last_time = time()
        self.total_time = 0.0

    def residual(self, param_values: Sequence) -> NDArray[np.float64]:
        """Compute the residual for the given parameters and data.

        Parameters
        ----------
        param_values
            The values of the parameters to update.
        """
        if self.multi_params:
            return self._calculate_multi(param_values)
        else:
            return self._calculate_single(param_values)

    def log(self, param_values, convergence: float = None):
        """Log the current fitting iteration and parameters.
        
        Parameters
        ----------
        param_values : Sequence
            The current values of the parameters.
        convergence : float, optional
            The convergence value of the fitting routine. 0.0 means
            no convergence, 1.0 means perfect convergence.
            If not provided, it will not be logged.
        """
        self.iteration += 1
        now = time()
        iter_time = now - self.last_time
        self.last_time = now
        self.total_time += iter_time

        log_message = f"Iteration {self.iteration}\n" \
                      f"----------{'-' * len(str(self.iteration))}\n" \
                      f"Best Parameters:\n"
        if self.update_keys is not None:
            update_params = _build_params_from_flat(
                self.update_keys, param_values)
            if self.multi_params is not None:
                for multi_param in self.multi_params:
                    multi_param_list = _extract_flat_value(
                        update_params, multi_param)
                    labels = (self.multi_params_labels or [
                        str(i) for i in range(len(multi_param_list))])
                    _update_flat_value(update_params, multi_param,
                                       dict(zip(labels, multi_param_list)))
            log_message += pformat(update_params) + "\n"
        else:
            log_message += pformat(param_values) + "\n"
        if convergence is not None:
            log_message += f"Convergence: {convergence:.5f}\n\n"
    
        log_message += "Iteration Runtime: "
        log_message += _get_hour_minute_second_string(iter_time) + "\n"
        log_message += "Total Runtime: "
        log_message += _get_hour_minute_second_string(self.total_time) + "\n\n"

        if self.print_log:
            print(log_message)
        if self.save_path is not None:
            path = Path(self.save_path) / f"{self.save_label}.log"
            mode = 'w' if self.iteration == 1 else 'a'
            with open(path, mode) as log_file:
                log_file.write(log_message)

    def _calculate_single(self, param_values):
        params = deepcopy(self.init_params)
        for key, value in zip(self.param_keys, param_values):
            _update_flat_value(params, key, value)
        cond = self._build_obj(easy_params(params))
        y_fits, y_values = [], []
        for i, (x_data, y_data) in enumerate(
                zip(self.x_data_sets, self.y_data_sets)):
            y_fit = self._calculate_dataset(
                cond, x_data, y_data, self.x_shift[i],
                self.x_normalize[i], self.y_shift[i], self.y_normalize[i])
            y_fit = self.postprocess(x_data, y_fit)
            y_fits.append(np.concatenate(list(y_fit.values())))
            y_values.append(np.concatenate(list(y_data.values())))
        return self.loss(np.concatenate(y_fits), np.concatenate(y_values))

    def _calculate_dataset(self, cond, x_data, y_data, x_shift,
                           x_normalize, y_shift, y_normalize):
        name, y_label_i, y_label_j = _get_label_indices(y_data.keys())
        y_fit = {label: np.zeros_like(y) for label, y in y_data.items()}
        n_points = len(next(iter(x_data.values())))
        for i in range(n_points):
            x = {label: x[i] for label, x in x_data.items()}
            y = _calc_y(cond, x, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label][i] = y[label]

        if x_shift is not None:
            y0 = _calc_y(cond, x_shift, y_data, name, y_label_i, y_label_j)
            for label in y0:
                y_fit[label] -= y0[label]
            if y_shift is not None:
                for label in y_fit:
                    y_fit[label] += y_shift[label]
        if x_normalize is not None:
            y0 = _calc_y(cond, x_normalize, y_data,
                         name, y_label_i, y_label_j)
            for label in y0:
                y_fit[label] /= y0[label]
            if y_normalize is not None:
                for label in y_fit:
                    y_fit[label] *= y_normalize[label]
        return y_fit
    
    def _calculate_multi(self, param_values):
        n_data_sets = len(next(iter(self.x_data.values())))
        total_loss = 0.0
        name, y_label_i, y_label_j = _get_label_indices(self.y_data.keys())
        params = deepcopy(self.init_params)
        for key, value in zip(self.param_keys, param_values):
            _update_flat_value(params, key, value)
        for i in range(n_data_sets):
            params_data_set = deepcopy(params)
            y_fit = {label: np.zeros_like(y[i])
                     for label, y in self.y_data.items()}
            for multi_param in self.multi_params:
                _update_flat_value(params_data_set, multi_param,
                                   _extract_flat_value(params, multi_param)[i])
            cond = self._build_obj(easy_params(params_data_set))
            for j in range(len(next(iter(y_fit.values())))):
                x = {label: x[i][j] for label, x in self.x_data.items()}
                y = _calc_y(cond, x, self.y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label][j] = y[label]
            if self.x_shift[0] is not None:
                x = {label: value[i] for label, value in self.x_shift[0].items()}
                y0 = _calc_y(cond, x, self.y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label] -= y0[label]
                if self.y_shift[0] is not None:
                    for label in y_fit:
                        y_fit[label] += self.y_shift[0][label][i]
            if self.x_normalize[0] is not None:
                x = {label: value[i]
                     for label, value in self.x_normalize[0].items()}
                y0 = _calc_y(cond, x, self.y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label] /= y0[label]
                if self.y_normalize[0] is not None:
                    for label in y_fit:
                        y_fit[label] *= self.y_normalize[0][label][i]
            x_data_set = {label: x[i] for label, x in self.x_data.items()}
            y_data_set = {label: y[i] for label, y in self.y_data.items()}
            y_fit = self.postprocess(x_data_set, y_fit)
            y_fit = np.concatenate(list(y_fit.values()))
            y_data_set = np.concatenate(list(y_data_set.values()))
            total_loss += self.loss(y_fit, y_data_set) / n_data_sets
        return total_loss

    def _build_obj(self, params):
        band = BandStructure(**params)
        band.discretize()
        cond = Conductivity(band, **params)
        return cond


class FullScatteringFitter:
    """This convenience class sets up a routine for full scattering
    rate fitting.

    Handles caching the residual and jacobian calculations to avoid
    redundant calculations.
    
    Attributes
    ----------
    scattering_rates : NDArray[np.float64]
        The current scattering rates.
    iteration : int
        The current iteration number.
    last_time : float
        The timestamp of the last fitting iteration.
    total_time : float
        The total time spent on the fitting routine.
    """
    def __init__(
            self, init_params: Mapping,
            x_data: Mapping | Sequence[Mapping],
            y_data: Mapping | Sequence[Mapping],
            x_shift: Mapping | Sequence[Mapping] = None,
            x_normalize: Mapping | Sequence[Mapping] = None,
            y_shift: Mapping | Sequence[Mapping] = None,
            y_normalize: Mapping | Sequence[Mapping] = None,
            regularization: Real = 0.0, peak_percentile: Real = 80,
            save_path: str = None, save_label: str = "fit",
            print_log: bool = True):
        self.params = easy_params(init_params)
        self.band = BandStructure(**self.params)
        self.band.discretize()

        self.x_data_sets = _normalize_data_sets(x_data, "x_data")
        self.y_data_sets = _normalize_data_sets(y_data, "y_data")
        _validate_data_sets(self.x_data_sets, self.y_data_sets)
        self.x_data = self.x_data_sets[0]
        self.y_data = self.y_data_sets[0]
        self.x_shift = _normalize_optional_data_sets(
            x_shift, len(self.x_data_sets), "x_shift")
        self.x_normalize = _normalize_optional_data_sets(
            x_normalize, len(self.x_data_sets), "x_normalize")
        self.y_shift = _normalize_optional_data_sets(
            y_shift, len(self.x_data_sets), "y_shift")
        self.y_normalize = _normalize_optional_data_sets(
            y_normalize, len(self.x_data_sets), "y_normalize")

        self.save_path = save_path
        self.save_label = save_label
        self.print_log = print_log
        self.iteration = 0
        self.last_time = time()
        self.total_time = 0.0

        self._scat_cache = None
        self.residual_vector = None
        self.jacobian_matrix = None

        self.regularization = regularization
        self.peak_percentile = peak_percentile
        if regularization > 0.0:
            self.grad_op = _calc_gradient_operator(self.band.surfaces[0.0])

    def residual(self, scattering_rates):
        self._calculate_shared(scattering_rates)
        if self.regularization > 0.0:
            return np.concatenate((self.residual_vector, self.reg_residual))
        else:
            return self.residual_vector

    def jacobian(self, scattering_rates):
        self._calculate_shared(scattering_rates)
        if self.regularization > 0.0:
            return scipy.sparse.vstack((
                self.jacobian_matrix,
                (self.regularization/10000 * self.grad_op)))
        else:
            return self.jacobian_matrix

    def reweigh_regularization(self, scattering_rates):
        """Reweigh the regularization term based on the
        current scattering rates.
        
        This is useful for preserving sharp features in the scattering
        rates while still smoothing out noise.

        Parameters
        ----------
        scattering_rates : NDArray[np.float64]
            The current scattering rates.
        """
        grad_applied = self.grad_op @ scattering_rates
        grad_norm = np.linalg.norm(grad_applied.reshape(-1, 3), axis=1)
        sharpness = np.percentile(grad_norm, self.peak_percentile)
        weight_triangle = 1.0 / np.sqrt(1.0 + (grad_norm/sharpness)**2)
        weight_rows = np.repeat(weight_triangle, 3)
        self.grad_op = scipy.sparse.diags(weight_rows) @ self.grad_op

    def log(self, intermediate_result: scipy.optimize.OptimizeResult):
        self.iteration += 1
        now = time()
        iter_time = now - self.last_time
        self.last_time = now
        self.total_time += iter_time

        log_message = f"Iteration {self.iteration}\n" \
                      f"----------{'-' * len(str(self.iteration))}\n" \
                      f"Best Scattering Rate Statistics (in THz):\n"
        scattering_rates = intermediate_result.x
        x_min, x_max = np.min(scattering_rates), np.max(scattering_rates)
        x_mean, x_std = np.mean(scattering_rates), np.std(scattering_rates)
        log_message += f"Mean: {x_mean:.4g}, Std: {x_std:.4g}, " \
                       f"Min: {x_min:.4g}, Max: {x_max:.4g}\n"

        log_message += "\nIteration Runtime: "
        log_message += _get_hour_minute_second_string(iter_time) + "\n"
        log_message += "Total Runtime: "
        log_message += _get_hour_minute_second_string(self.total_time) + "\n\n"

        if self.print_log:
            print(log_message)
        if self.save_path is not None:
            path = Path(self.save_path) / f"{self.save_label}.log"
            mode = 'w' if self.iteration == 1 else 'a'
            with open(path, mode) as log_file:
                log_file.write(log_message)
    
    def _calculate_shared(self, scattering_rates):
        if self._scat_cache is not None and np.array_equal(
                self._scat_cache, scattering_rates):
            return

        if self.band.symmetry is None:
            scat_full = scattering_rates
        else:
            scat_full = self.band.symmetry_projector.T @ scattering_rates
        cond = Conductivity(
            self.band, scattering_rate=scat_full, **self.params)
        self.y_fit = []
        self.jacobian_matrix = []
        for i in range(len(self.x_data_sets)):
            self._calc_y_multiplier(cond, i)
            self._calc_fit(scat_full, len(scattering_rates), i)
            self._apply_y_shifts(cond, i)
        if self.regularization > 0.0:
            self.reg_residual = \
                self.regularization/10000 * self.grad_op @ scattering_rates

        y_fit = np.concatenate([
            np.concatenate(list(dataset.values())) for dataset in self.y_fit])
        y_data = np.concatenate([
            np.concatenate(list(dataset.values()))
            for dataset in self.y_data_sets])
        self.residual_vector = y_fit - y_data
        self.jacobian_matrix = np.vstack([
            np.vstack(list(dataset.values()))
            for dataset in self.jacobian_matrix])
        self._scat_cache = np.copy(scattering_rates)

    def _calc_y_multiplier(self, cond, dataset_index):
        y_data = self.y_data_sets[dataset_index]
        if self.x_normalize[dataset_index] is not None:
            y_name, y_label_i, y_label_j = _get_label_indices(y_data.keys())
            x_normalize = self.x_normalize[dataset_index]
            self.y0 = _calc_y(cond, x_normalize, y_data,
                              y_name, y_label_i, y_label_j)
            transpose_solution = cond._factorization.solve(
                cond._finite_elements[0.0].vhat_projections, trans='T')
            self.j0 = self._calc_jac(
                cond._finite_elements[0.0].jacobian_sums,
                cond._linear_solution, transpose_solution, cond,
                np.linalg.inv(cond.sigma))
            self._y_multiply = {label: 1.0 / self.y0[label]
                                for label in self.y0}
        else:
            self._y_multiply = {label: 1.0 for label in y_data}
        if self.y_normalize[dataset_index] is not None:
            for label in self._y_multiply:
                self._y_multiply[label] *= \
                    self.y_normalize[dataset_index][label]

    def _apply_y_shifts(self, cond, dataset_index):
        y_data = self.y_data_sets[dataset_index]
        y_name, y_label_i, y_label_j = _get_label_indices(y_data.keys())
        if self.x_shift[dataset_index] is not None:
            y0 = _calc_y(cond, self.x_shift[dataset_index], y_data,
                            y_name, y_label_i, y_label_j)
            for label in self.y_fit[dataset_index]:
                self.y_fit[dataset_index][label] -= y0[label]
        if self.y_shift[dataset_index] is not None:
            for label in self.y_fit[dataset_index]:
                self.y_fit[dataset_index][label] += \
                    self.y_shift[dataset_index][label]

    def _calc_fit(self, scattering_rates, n_irr_scat=None,
                  dataset_index=0):
        x_data = self.x_data_sets[dataset_index]
        y_data = self.y_data_sets[dataset_index]
        y_name, y_label_i, y_label_j = _get_label_indices(y_data.keys())
        n_data = len(next(iter(x_data.values())))
        if n_irr_scat is None:
            n_irr_scat = len(scattering_rates)
        y_fit = {label: np.empty_like(y) for label, y in y_data.items()}
        jacobian_matrix = {
            label: np.empty((len(y), n_irr_scat))
            for label, y in y_data.items()}
        results = [self._calc_single_data_point(scattering_rates, i,
                                                dataset_index)
                   for i in range(n_data)]
        for i, r in enumerate(results):
            rho, jac = r
            for label in y_data:
                a, b = y_label_i[label], y_label_j[label]
                y_fit[label][i] = self._y_multiply[label] * rho[a, b]
                jacobian_matrix[label][i] = self._y_multiply[label] * THz * jac[a, b]
                if self.x_normalize[dataset_index] is not None:
                    jacobian_matrix[label][i] -= \
                        self._y_multiply[label] * THz * \
                        rho[a, b]/self.y0[label] * self.j0[a, b]
        self.y_fit.append(y_fit)
        self.jacobian_matrix.append(jacobian_matrix)

    def _calc_single_data_point(self, scattering_rates, index, dataset_index=0):
        cond = Conductivity(self.band, scattering_rate=scattering_rates,
                            **self.params)
        x_data = self.x_data_sets[dataset_index]
        x = {label: x[index] for label, x in x_data.items()}
        for label, value in x.items():
            setattr(cond, label, value)
        cond.calculate()
        rho = np.linalg.inv(cond.sigma)

        transpose_solution = cond._factorization.solve(
            cond._finite_elements[0.0].vhat_projections, trans='T')
        jac = self._calc_jac(
            cond._finite_elements[0.0].jacobian_sums,
            cond._linear_solution, transpose_solution, cond, rho)
        return rho, jac

    def _calc_jac(self, alpha, X, Y, cond, rho):
        if cond.band.periodic:
            X = cond.band.periodic_projector.T @ X
            Y = cond.band.periodic_projector.T @ Y
        # "matrix-like" parts of Y^T dGamma/dgamma X
        term_1 = np.array([
            [alpha.T @ (Y[:, a] * X[:, b]) for b in range(X.shape[1])]
             for a in range(Y.shape[1])])
        # Y_pa sum_i alpha_pj X_jb
        alpha_X = alpha @ X
        term_2 = np.einsum("pa,pb->abp", Y, alpha_X)
        # X_pb sum_i alpha_pi Y_ia
        alpha_Y = alpha @ Y
        term_3 = np.einsum("pb,pa->abp", X, alpha_Y)
        jac = term_1 + term_2 + term_3

        # "tensor-like" term of Y^T dGamma/dgamma X
        i_idx = self.band.kfaces[:, 0]
        j_idx = self.band.kfaces[:, 1]
        k_idx = self.band.kfaces[:, 2]
        for perm in [(i_idx, j_idx, k_idx), (j_idx, k_idx, i_idx),
                     (k_idx, i_idx, j_idx), (j_idx, i_idx, k_idx),
                     (i_idx, k_idx, j_idx), (k_idx, j_idx, i_idx)]:
            i, j, p = perm
            # sum_ij (alpha_ijp/2) Y_ia X_jb
            X_vec = X[j, :]
            Y_vec = Y[i, :]
            for a in range(Y_vec.shape[1]):
                for b in range(X_vec.shape[1]):
                    np.add.at(
                        jac[a, b], p, (cond._finite_elements[0.0].jacobians
                                       * Y_vec[:, a] * X_vec[:, b] / 2))
        jac = np.einsum("ai,ijp,jb->abp", rho, jac, rho)
        # chain rule for drho/d(1/tau) = (1/v) drho/dgamma
        jac /= cond._finite_elements[0.0].vmags
        jac *= e**2 / (4 * np.pi**3 * hbar) / self.band.bz_ratio / 60
        if cond.band.symmetry is not None:
            jac_full = jac
            jac = np.empty((*jac.shape[:-1],
                            cond.band.symmetry_projector.shape[0]))
            for a in range(jac.shape[0]):
                for b in range(jac.shape[1]):
                    jac[a, b] = cond.band.symmetry_projector @ jac_full[a, b]
        return jac


def _calc_gradient_operator(surface: IsoEnergySurface):
    triangle_points = surface.kpoints[surface.kfaces] / angstrom
    jacobians = np.linalg.norm(
        np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                 triangle_points[:, 2] - triangle_points[:, 0]),
        axis=-1)
    diffs = triangle_points - np.roll(triangle_points, -2, axis=1)
    weights = np.sqrt(2*jacobians)
    weighted_diffs = diffs / weights[:, np.newaxis, np.newaxis]
    # Reorder (faces, vertices, dims) -> (faces, dims, vertices) and flatten.
    # For each (face, dim), lists the 3 values for vertices [j, k, i].
    weighted_diffs = weighted_diffs.transpose(0, 2, 1).flatten()

    j_idx = np.roll(surface.kfaces, -1, axis=1)
    n_faces = len(surface.kfaces)
    rows = np.repeat(np.arange(3*n_faces), 3)
    cols = np.tile(j_idx, (1, 3)).ravel()

    grad_op = scipy.sparse.csr_matrix((weighted_diffs, (rows, cols)),
                                      shape=(3*n_faces, len(surface.kpoints)))
    if surface.symmetry_projector is not None:
        grad_op = grad_op @ surface.symmetry_projector.T
    return grad_op


def _mean_squared_error(y_fit, y_data):
    """Mean squared error loss function."""
    return np.mean((y_fit - y_data) ** 2)


def _mean_absolute_error(y_fit, y_data):
    """Mean absolute error loss function."""
    return np.mean(np.abs(y_fit - y_data))


def _dummy_processor(x, y):
    """A dummy processing function that does nothing.
    Exists because lambdas cannot be pickled, and so cannot be used in
    multiprocessed fitting routines."""
    return y


def fit_model(x_data: Mapping | Sequence[Mapping],
              y_data: Mapping | Sequence[Mapping],
              init_params: Mapping, bounds: Mapping,
              multi_params: Collection[str] = None,
              multi_params_labels: Collection[str] = None,
              x_shift: Mapping | Sequence[Mapping] = None,
              x_normalize: Mapping | Sequence[Mapping] = None,
              y_shift: Mapping | Sequence[Mapping] = None,
              y_normalize: Mapping | Sequence[Mapping] = None,
              optimizer: Callable = scipy.optimize.differential_evolution,
              loss: Callable = _mean_absolute_error,
              postprocess: Callable = _dummy_processor,
              save_path: str = None, save_label: str = None,
              print_log: bool = True, **kwargs):
    """Convenience function to set up and run a fitting routine.

    This uses the `optimizer` (assumed to behave like a SciPy optimizer)
    to perform a global fit. The optimizer is hard-coded, because the
    callback functions are highly specific to each optimizer.
    Saves the results to the specified path.

    Parameters
    ----------
    x_data
        The independent variable data (e.g. field). The name of the
        variable is mapped to the data, e.g.
        ``{'field': [0, 1, 2]}``. In case of nonempty ``multi_params``,
        the value must be a collection of sequences, where each sequence
        corresponds to a different parameter to be fitted,
        e.g. ``{'field': [[[0.5, 1.5, 2.5], [0.6, 1.6, 2.6]]}``.
        Alternatively, a sequence of mappings can be provided to combine
        multiple datasets with shared parameters. The corresponding
        ``x_data`` and ``y_data`` mappings are paired by position, and
        each dataset may contain a different number of points.
    y_data
        The dependent variable data (e.g. conductivity). The name of
        the variable is mapped to the data, e.g.
        ``{'sigma_xx': [1.1, 2.4, 3.8]}``. The name of each variable
        must start with "sigma" or "rho" (for conductivity or
        resistivity, respectively), and you can add a suffix to specify
        the component (e.g. ``'sigma_xx'``, ``'rho_xy'``). In case of
        nonempty ``multi_params``, the value must be a collection of
        sequences, where each sequence corresponds to a different
        parameter to be fitted, e.g. ``{'rho_zz': [[1.1, 2.4, 3.8],
        [1.2, 2.5, 3.9]]}``.
        Alternatively, a sequence of mappings can be provided to combine
        multiple datasets with shared parameters. The loss is calculated
        using the concatenated processed values from all datasets.
    init_params
        Initial parameters for the fitting routine, and also other
        parameters for initiallizing the classes. This is passed through
        ``easy_params`` to the ``BandStructure`` and ``Conductivity``.
    bounds
        Bounds for the fitting parameters. This mapping has the same
        structure as ``init_params``, but only containing the variables
        that are to be fitted, and their values in the mapping must be
        a collection of the form (min, max).
    multi_params
        A collection of parameters that are to be fitted differently for
        the different datasets in ``x_data`` and ``y_data``, if there is
        more than one. To make it precise, each label must be a
        dot-separated string, showing the "path" to the value in the
        parameters dictionary, e.g. ``"band_params.mu"`` or
        ``"scattering_params.nu.0"``. These parameters must themselves
        be collections in the parameters dictionary, showing the value
        for every dataset e.g.
        ``{'band_params': {'mu': [0.1, 0.2, 0.3]}}`` in ``init_params``
        or ``{'band_params': {'mu': [(0.1, 0.9), (0.2, 0.8),
        (0.3, 0.7)]}}`` in ``bounds``.
    multi_params_labels
        A collection of labels for the different datasets in ``x_data``
        and ``y_data``. The output of fits that contain multi-parameters
        will be saved in separate files for each dataset, and the labels
        will be appended to the ``save_label`` with an underscore. If
        not provided, the datasets will be labeled with their index
        in the collection, e.g. ``"fit_label_0.json"``.
    x_shift
        If provided, the y values will be shifted by the y value at
        this x point. The mapping must have the same structure as
        ``x_data``, but with single values instead of arrays as
        the values. For multiple datasets, provide one such mapping per
        dataset.
    x_normalize
        If provided, the y values will be normalized by the y value
        at this x point. The mapping must have the same structure
        as ``x_data``, but with single values instead of arrays as
        the values. Note that shifts are applied before normalization.
    y_shift
        The y values will be shifted to this value (if ``x_shift`` is
        provided). The mapping must have the same structure as
        ``y_data``, but with single values instead of arrays as the
        values.
    y_normalize
        The y values will be normalized to this value (if
        ``x_normalize`` is provided). The mapping must have the same
        structure as ``y_data``, but with single values instead of
        arrays as the values. For multiple datasets, provide one such
        mapping per dataset.
    optimizer
        The optimizer function to use for fitting. It must have the
        same interface as the SciPy optimizers.
    loss
        A function that takes the fit and data y values, and returns a
        scalar loss value. By default, the mean absolute error is used.
    postprocess
        This callable is applied to the data y values before
        calculating the loss. It takes ``x_data`` and a ``y`` with a
        format similar to ``y_data``, and returns the processed ``y``,
        again with a format similar to ``y_data``. By default, no
        postprocessing is applied. An example use case is applying a
        mask to ignore parts of the values where the data can be
        unreliable. This is to apply the same preprocessing that might
        be applied to the data before fitting.
    save_path
        The directory where the fitting results will be saved.
        If not provided, results will not be saved.
    save_label
        Label of the results. If not provided, will be set to
        ``f"y_label_x_label"``. If ``y_label` or ``x_label`` are
        collections of string, they will be joined with an underscore.
    print_log
        If True, the fitting progress will be printed to the console.
    **kwargs
        Additional keyword arguments passed to the `optimizer`
    """
    x_data_sets = _normalize_data_sets(x_data, "x_data")
    y_data_sets = _normalize_data_sets(y_data, "y_data")
    _validate_data_sets(x_data_sets, y_data_sets)
    if multi_params and len(x_data_sets) > 1:
        raise ValueError(
            "Combining dataset mappings with multi_params is not supported; "
            "use the existing nested mapping format.")
    if save_label is None:
        x_labels = list(x_data_sets[0])
        y_labels = list(y_data_sets[0])
        x_string = x_labels[0] if len(x_labels) == 1 else "_".join(x_labels)
        y_string = y_labels[0] if len(y_labels) == 1 else "_".join(y_labels)
        save_label = f"{y_string}_{x_string}"

    update_keys = _extract_flat_keys(bounds, bounds=True)
    bounds = [_extract_flat_value(bounds, key) for key in update_keys]
    x0 = [_extract_flat_value(init_params, key) for key in update_keys]
    if multi_params:
        update_keys, bounds, x0, init_params = _multiply_multi_params(
            update_keys, bounds, x0, init_params,
            multi_params, len(next(iter(x_data_sets[0].values()))))
    for i in range(len(update_keys)):
        if x0[i] is None:
            x_min, x_max = bounds[i]
            x0[i] = (x_min + x_max) / 2

    begin_time = datetime.now()
    fitter = FittingRoutine(
        init_params, x_data, y_data, x_shift, x_normalize,
        y_shift, y_normalize, loss, postprocess,
        save_path, save_label, update_keys=update_keys,
        multi_params=multi_params, multi_params_labels=multi_params_labels,
        print_log=print_log)
    result = optimizer(fitter.residual, bounds=bounds, x0=x0,
                       callback=fitter.log, **kwargs)
    end_time = datetime.now()

    print(result.message)
    print("\n\n")
    return _save_fit_result(
        result, init_params, update_keys, begin_time, end_time,
        save_path, save_label, multi_params=multi_params,
        multi_params_labels=multi_params_labels)


def fit_full_scattering(
        x_data: Mapping | Sequence[Mapping],
        y_data: Mapping | Sequence[Mapping],
        init_params: Mapping, init_scattering: Real = 10.0,
        bounds: Sequence[Sequence] | Sequence = (0, np.inf),
        regularization: Sequence[Real] | Real = 0.0,
        peak_percentile: Real = 80,
        x_shift: Mapping | Sequence[Mapping] = None,
        x_normalize: Mapping | Sequence[Mapping] = None,
        y_shift: Mapping | Sequence[Mapping] = None,
        y_normalize: Mapping | Sequence[Mapping] = None,
        save_path: str = None, save_label: str = None,
        print_log: bool = True, **kwargs):
    """Convenience function to set up and run a fitting routine.

    This uses the `optimizer` (assumed to behave like a SciPy optimizer)
    to perform a global fit. The optimizer is hard-coded, because the
    callback functions are highly specific to each optimizer.
    Saves the results to the specified path.

    Parameters
    ----------
    x_data
        The independent variable data (e.g. field). The name of the
        variable is mapped to the data, e.g.
        ``{'field': [0, 1, 2]}``.
        Alternatively, a sequence of mappings can be provided to combine
        multiple datasets with shared parameters. The corresponding
        ``x_data`` and ``y_data`` mappings are paired by position, and
        each dataset may contain a different number of points.
    y_data
        The dependent variable data (e.g. conductivity). The name of
        the variable is mapped to the data, e.g.
        ``{'sigma_xx': [1.1, 2.4, 3.8]}``. The name of each variable
        must start with "sigma" or "rho" (for conductivity or
        resistivity, respectively), and you can add a suffix to specify
        the component (e.g. ``'sigma_xx'``, ``'rho_xy'``).
        Alternatively, a sequence of mappings can be provided to combine
        multiple datasets with shared parameters. The residual and
        Jacobian are constructed from all datasets.
    init_params
        Parameters for initiallizing the classes. This is passed through
        ``easy_params`` to the ``BandStructure`` and ``Conductivity``.
    init_scattering
        Initial scattering rate. Can either be: a single value;
        a function of kx, ky, kz, vx, vy, vz, energy (or E),
        and temperature (or T), note that all of these parameters are
        optional and the function can take any subset of them, as long
        as the names match, and also the function must collect extra
        keyword arguments (``**kwargs``) to avoid errors; a mapping
        of a mapping of the form
        ``{'scattering_model: {...}, 'scattering_params: {...}}``
        which is the same as `elecboltz.params.easy_params` accepts;
        or an array of real numbers which should match the number of
        points in the discretization of the Fermi ``surface`` of the
        `elecboltz.bandstructure.BandStructure` model made from the
        parameters.
    bounds
        Bounds for the scattering rates. This would typically be a
        single sequence of the form (min, max) to set the boundaries
        of all scattering rates, but it can also be a sequence of
        sequences, to set the individual boundaries of each scattering
        rate of each ``kpoint`` of the band structure.
    x_shift
        If provided, the y values will be shifted by the y value at
        this x point. The mapping must have the same structure as
        ``x_data``, but with single values instead of arrays as
        the values.
    x_normalize
        If provided, the y values will be normalized by the y value
        at this x point. The mapping must have the same structure
        as ``x_data``, but with single values instead of arrays as
        the values. Note that shifts are applied before normalization.
    y_shift
        The y values will be shifted to this value (if ``x_shift`` is
        provided). The mapping must have the same structure as
        ``y_data``, but with single values instead of arrays as the
        values.
    y_normalize
        The y values will be normalized to this value (if
        ``x_normalize`` is provided). The mapping must have the same
        structure as ``y_data``, but with single values instead of
        arrays as the values.
    regularization
        The regularization parameter(s) to use. If multiple values are
        provided, multiple fits will be performed sequentially with
        the results of the previous fit used as the initial guess for
        the next fit. Each step, the regularization is weighted to
        preserve the sharp features of the last step, while
        smoothing out the noise. Therefore, repeating a step
        will actually change the result, and it is recommended
        to repeat the last step.
    peak_percentile
        The percentile of the gradient norm that is assumed to be due
        to the "peaks" in the scattering rate. Used to reweigh the
        regularization term to preserve sharp features in the scattering
        rates while still smoothing out noise.
    loss
        A function that takes the fit and data y values, and returns a
        scalar loss value. By default, the mean absolute error is used.
    save_path
        The directory where the fitting results will be saved.
        If not provided, results will not be saved.
    save_label
        Label of the results. If not provided, will be set to
        ``f"y_label_x_label"``. If ``y_label` or ``x_label`` are
        collections of string, they will be joined with an underscore.
    print_log
        If True, the fitting progress will be printed to the console.
    **kwargs
        Additional keyword arguments passed to the `optimizer`
    """
    x_data_sets = _normalize_data_sets(x_data, "x_data")
    y_data_sets = _normalize_data_sets(y_data, "y_data")
    _validate_data_sets(x_data_sets, y_data_sets)
    if save_label is None:
        x_labels = list(x_data_sets[0])
        y_labels = list(y_data_sets[0])
        x_string = x_labels[0] if len(x_labels) == 1 else "_".join(x_labels)
        y_string = y_labels[0] if len(y_labels) == 1 else "_".join(y_labels)
        save_label = f"{y_string}_{x_string}"
    begin_time = datetime.now()

    reg = (regularization[0] if isinstance(regularization, Sequence)
           else regularization)
    fitter = FullScatteringFitter(
        init_params, x_data, y_data, x_shift, x_normalize,
        y_shift, y_normalize, reg, peak_percentile,
        save_path, save_label, print_log)
    x0 = _get_init_scatrate_fullfit(fitter.band, init_scattering)
    for tol in ['ftol', 'xtol', 'gtol']:
        if tol not in kwargs:
            kwargs[tol] = 1e-6

    result = scipy.optimize.least_squares(
        fitter.residual, jac=fitter.jacobian, x0=x0, bounds=bounds,
        x_scale='jac', callback=fitter.log, **kwargs)
    if isinstance(regularization, Sequence):
        for reg in regularization[1:]:
            print(result.message + "\n\n")
            if save_path is not None:
                path = Path(save_path) / f"{save_label}.log"
                with open(path, 'a') as log_file:
                    log_file.write(result.message + "\n\n")
            fitter.reweigh_regularization(result.x)
            fitter.regularization = reg
            result = scipy.optimize.least_squares(
                fitter.residual, jac=fitter.jacobian, x0=result.x,
                bounds=bounds, x_scale='jac', callback=fitter.log, **kwargs)

    end_time = datetime.now()
    print(result.message)
    print("\n\n")
    _save_full_scattering_result(
        result, fitter, init_params, init_scattering,
        begin_time, end_time, save_path, save_label)
    return result


def _get_init_scatrate_fullfit(band, init_scattering):
    surface = band.surfaces[0.0]
    if band.symmetry is None:
        n_points = len(surface.kpoints)
    else:
        n_points = surface.n_irreducible

    if isinstance(init_scattering, Real):
        return np.full(n_points, init_scattering)
    else:
        if isinstance(init_scattering, Callable):
            func = init_scattering
        elif isinstance(init_scattering, Mapping):
            func = build_scattering_function(**init_scattering)
        else:
            raise ValueError("Invalid type for init_scattering. Must be a "
                             "real number, callable function, or mapping.")
        kx, ky, kz = surface.kpoints[:n_points].T
        vx, vy, vz = band.velocity_func(kx, ky, kz)
        energy = 0
        temperature = 0
        return np.array(func(
            kx=kx, ky=ky, kz=kz, vx=vx, vy=vy, vz=vz,
            energy=energy, E=energy, temperature=temperature, T=temperature))
    

def _get_label_indices(labels: Collection[str]):
    """Extract the names and indices from y_data keys."""
    name, i, j = {}, {}, {}
    for label in labels:
        name[label], index_labels = label.split("_")
        i[label] = {'x': 0, 'y': 1, 'z': 2}[index_labels[0]]
        j[label] = {'x': 0, 'y': 1, 'z': 2}[index_labels[1]]
    return name, i, j


def _normalize_data_sets(data, name):
    if isinstance(data, Mapping):
        return (data,)
    if (isinstance(data, Sequence) and not isinstance(data, (str, bytes))
            and all(isinstance(item, Mapping) for item in data)):
        if not data:
            raise ValueError(f"{name} must contain at least one dataset.")
        return tuple(data)
    raise TypeError(f"{name} must be a mapping or a sequence of mappings.")


def _normalize_optional_data_sets(data, n_data_sets, name):
    if data is None:
        return [None] * n_data_sets
    if isinstance(data, Mapping):
        data_sets = (data,)
    elif (isinstance(data, Sequence)
          and not isinstance(data, (str, bytes))):
        if not data:
            raise ValueError(f"{name} must contain at least one dataset.")
        if any(item is not None and not isinstance(item, Mapping)
               for item in data):
            raise TypeError(
                f"{name} must be a mapping, None, or a sequence of "
                "mappings and None values.")
        data_sets = tuple(data)
    else:
        raise TypeError(
            f"{name} must be a mapping, None, or a sequence of mappings "
            "and None values.")
    if len(data_sets) == 1 and n_data_sets > 1:
        raise ValueError(
            f"{name} must contain one mapping or None for each dataset.")
    if len(data_sets) != n_data_sets:
        raise ValueError(
            f"{name} must contain {n_data_sets} dataset mappings or None "
            "values.")
    return list(data_sets)


def _validate_data_sets(x_data_sets, y_data_sets):
    if len(x_data_sets) != len(y_data_sets):
        raise ValueError("x_data and y_data must contain the same number "
                         "of datasets.")


def _calc_y(cond, x_data, y_data, name, y_label_i, y_label_j):
    y = {}
    for label, x in x_data.items():
        setattr(cond, label, x)
    if 'rho' in name.values():
        cond.calculate()
        rho = np.linalg.inv(cond.sigma)
    else:
        cond.calculate(sorted(set(y_label_i.values())),
                       sorted(set(y_label_j.values())))
    for label in y_data:
        if name[label] == 'sigma':
            y[label] = cond.sigma[y_label_i[label], y_label_j[label]]
        elif name[label] == 'rho':
            y[label] = rho[y_label_i[label], y_label_j[label]]
        else:
            raise ValueError(f"Unknown y_data key: {name[label]}")
    return y


def _multiply_multi_params(update_keys, bounds, x0, init_params,
                           multi_params, n):
    """Expand multi-parameters in the update keys, bounds, and x0 lists."""
    i = 0
    while i < len(update_keys):
        if update_keys[i] in multi_params: # multi-parameter with single bound
            if not isinstance(x0[i], Sequence):
                init_param_list = []
            for j in range(n):
                update_keys.insert(i + j + 1, f"{update_keys[i]}.{j}")
                bounds.insert(i + j + 1, bounds[i])
                if isinstance(x0[i], Sequence):
                    x0.insert(i + j + 1, x0[i][j])
                else:
                    x0.insert(i + j + 1, x0[i])
                    init_param_list.append(x0[i])
            if not isinstance(x0[i], Sequence):
                _update_flat_value(init_params, update_keys[i],
                                   init_param_list)
            update_keys.pop(i)
            bounds.pop(i)
            x0.pop(i)
            i += n
        else:
            parent = '.'.join(update_keys[i].split('.')[:-1])
            if parent in multi_params and x0[i] is None:
                parent_value = _extract_flat_value(init_params, parent)
                if isinstance(parent_value, Sequence):
                    _update_flat_value(init_params, parent,
                                       parent_value + [parent_value[0]])
                    x0[i] = parent_value[0]
                elif parent_value is not None:
                    _update_flat_value(init_params, parent, [parent_value])
                    x0[i] = parent_value
            i += 1
    return update_keys, bounds, x0, init_params


def _extract_flat_keys(params, bounds=False):
    """Extract dots-separated keys from a nested structure.
    
    This function recursively extracts keys from a nested dictionary
    containing dictionaries and lists, and returns a flat list of keys.
    Each key is represented as a string with dot-separated keys and
    indices, e.g. ``"band_params.a"``, is ``params["band_params"]["a"]``
    or ``"scattering_params.nu.0"`` is
    ``params["scattering_params"]["nu"][0]``.

    Parameters
    ----------
    params : Mapping
        The nested dictionary from which to extract keys.
    
    Returns
    -------
    list[str]
        A list of flattened keys, where each key is a string
        representing the path to the value in the nested structure.
    """
    keys = []
    if isinstance(params, Mapping):
        for key in params:
            value = params[key]
            if _is_value_nested(value, bounds):
                for val_key in _extract_flat_keys(value, bounds=bounds):
                    keys.append(f"{key}.{val_key}")
            else:
                keys.append(key)
    elif isinstance(params, Sequence):
        for (i, value) in enumerate(params):
            if _is_value_nested(value, bounds):
                for val_key in _extract_flat_keys(value, bounds=bounds):
                    keys.append(f"{i}.{val_key}")
            else:
                keys.append(str(i))
    return keys


def _is_value_nested(value, bounds):
    if bounds:
        if isinstance(value, Mapping):
            return any(isinstance(v, Collection) and
                       not isinstance(v, str) for v in value.values())
        elif isinstance(value, Collection):
            return any(isinstance(v, Collection) and
                       not isinstance(v, str) for v in value)
    else:
        return isinstance(value, Collection) and not isinstance(value, str)


def _extract_flat_value(params: Mapping, flat_key: str):
    value = params
    key_parts = flat_key.split('.')
    while key_parts:
        key = key_parts.pop(0)
        if str.isnumeric(key) and not isinstance(value, Mapping):
            key = int(key)
            if isinstance(value, Sequence):
                if key >= len(value):
                    return None
            else:
                return None
        elif key not in value:
            return None
        value = value[key]
    return value


def _update_flat_value(params: Mapping, flat_key: str, value):
    level_params = params
    key_parts = flat_key.split('.')
    while len(key_parts) > 1:
        key = key_parts.pop(0)
        if str.isnumeric(key):
            key = int(key)
            if key >= len(level_params):
                return
        elif key not in level_params:
            return
        level_params = level_params[key]
    key = key_parts[0]
    if str.isnumeric(key):
        key = int(key)
        if key >= len(level_params):
            return
    level_params[key] = value


def _build_params_from_flat(param_keys, param_values):
    """Build a nested dictionary from flattened keys and values.

    Parameters
    ----------
    param_keys : Collection[str]
        The "flattened" keys of the parameters. See
        ``_extract_flat_keys`` for more information.
    param_values : Sequence
        The values corresponding to the keys in ``param_keys``.

    Returns
    -------
    dict
        A nested dictionary where the keys are the flattened keys
        and the values are the corresponding values from ``param_values``.
    """
    params = dict()
    param_values = list(param_values)
    for key in param_keys:
        key_parts = key.split('.')
        key = key_parts[0]
        level_params = params
        for part in key_parts[1:]:
            if str.isnumeric(part):
                part = int(part)
                if isinstance(level_params, dict):
                    if key not in level_params:
                        level_params[key] = []
                if isinstance(level_params, list):
                    if level_params[key] == dict():
                        level_params[key] = []
                while part >= len(level_params[key]):
                    level_params[key].append(dict())
            elif isinstance(level_params, dict) and key not in level_params:
                level_params[key] = dict()
            elif isinstance(level_params, list) and key >= len(level_params):
                level_params.append(dict())
            level_params = level_params[key]
            key = part
        level_params[key] = param_values.pop(0)
    return params


def _save_fit_result(result, init_params, update_keys, begin_time,
                     end_time, save_path, save_label,
                     multi_params=None, multi_params_labels=None):
    result = _result_to_serializable(result)
    result['fit_params'] = _build_params_from_flat(update_keys, result['x'])
    result['residual'] = result.pop('fun', None)
    result['evaluations'] = result.pop('nfev', None)
    result['iterations'] = result.pop('nit', None)
    result['jacobian'] = result.pop('jac', None)
    result.pop('x', None)

    result['init_params'] = _build_params_from_flat(
        update_keys, [_extract_flat_value(init_params, key)
                      for key in update_keys])
    all_keys = _extract_flat_keys(init_params, bounds=False)
    fixed_keys = set(all_keys) - set(update_keys)
    result['fixed_params'] = _build_params_from_flat(
        fixed_keys, [_extract_flat_value(init_params, key)
                     for key in fixed_keys])
    
    if 'fixed_filling' in init_params:
        _update_result_chemical_potential(result)

    result['begin_time'] = begin_time.isoformat()
    result['end_time'] = end_time.isoformat()
    result['runtime'] = (end_time - begin_time).total_seconds()

    if multi_params:
        # get length of multi-parameter values
        multi_param_length = len(_extract_flat_value(
            init_params, multi_params[0]))
        multi_results = {}
        for i in range(multi_param_length):
            multi_result = deepcopy(result)
            for multi_param in multi_params:
                value = _extract_flat_value(
                    result['fit_params'], multi_param)[i]
                _update_flat_value(
                    multi_result['fit_params'], multi_param, value)
            if multi_params_labels is not None:
                label = multi_params_labels[i]
            else:
                label = str(i)
            multi_results[label] = multi_result

    if save_path is not None:
        if multi_params:
            for multi_label, multi_result in multi_results.items():
                path = Path(save_path) / f"{save_label}_{multi_label}.json"
                with path.open('w') as f:
                    json.dump(multi_result, f, indent=2)
        else:
            path = Path(save_path) / f"{save_label}.json"
            with path.open('w') as f:
                json.dump(result, f, indent=2)
    return result


def _save_full_scattering_result(
        result, fitter, init_params, init_scattering,
        begin_time, end_time, save_path, save_label):
    if fitter.band.symmetry is not None:
        result['scattering_rate_irreducible'] = result.x
        result['scattering_rate'] = fitter.band.symmetry_projector.T @ result.x
        result['jacobian'] = (fitter.band.symmetry_projector.T
                              @ fitter.jacobian_matrix)
    else:
        result['scattering_rate'] = result.x
        result['jacobian'] = fitter.jacobian_matrix
    result.pop('x', None)

    result['iterations'] = result.pop('nit', None)
    result['evaluations'] = result.pop('nfev', None)
    result['jacobian_evaluations'] = result.pop('njev', None)
    result['init_params'] = init_params
    result['init_scattering'] = init_scattering

    result.pop('fun', None)
    result.pop('grad', None)
    result.pop('jac', None)
    result.pop('active_mask', None)

    result['begin_time'] = begin_time.isoformat()
    result['end_time'] = end_time.isoformat()
    result['runtime'] = (end_time - begin_time).total_seconds()

    result = _result_to_serializable(result)
    if save_path is not None:
        path = Path(save_path) / f"{save_label}.json"
        with path.open('w') as f:
            json.dump(result, f, indent=2)
    return result


def _update_result_chemical_potential(result):
    params = deepcopy(result['fit_params'])
    _deep_update(params, result['fixed_params'])
    band = BandStructure(**easy_params(params))
    band.discretize()
    if 'mu' in band.band_params:
        mu = band.band_params['mu']
        if 'energy_scale' in params['fixed_params']:
            mu /= params['fixed_params']['energy_scale']
        result['fit_params']['band_params']['mu'] = mu
        result['init_params']['band_params']['mu'] = \
            result['fixed_params']['band_params'].pop('mu')
        if len(result['fixed_params']['band_params']) == 0:
            result['fixed_params'].pop('band_params', None)
    else:
        result['fit_params']['chemical_potential'] = band.chemical_potential
        result['fixed_params'].pop('chemical_potential', None)


def _result_to_serializable(result):
    serializable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        else:
            try:
                json.dumps(value)  # test serializability
                serializable[key] = value
            except (TypeError, OverflowError):
                serializable[key] = str(value)
    return serializable


def _get_hour_minute_second_string(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    time_string = ""
    if hours > 0:
        time_string += f"{int(hours)} hours "
    if minutes > 0:
        time_string += f"{int(minutes)} minutes "
    time_string += f"{int(round(seconds))} seconds"
    return time_string
