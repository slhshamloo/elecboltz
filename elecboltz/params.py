from .scatrate import build_scattering_function
from .kernel import build_kernel
from copy import deepcopy
import json


def easy_params(params: dict) -> dict:
    """Convenience function to set parameters for the simulation.

    List of convenience features:

    * | Load parameters from the output of a fit from the path indicated
      | by the ``load_fit`` parameter.
    * | Set unit cell dimensions with named parameters ``a``, ``b``,
      | and ``c``. If ``b`` or ``c`` are not given, they are assumed
      | to be equal to ``a``.
    * Define an ``energy_scale`` which scales all energy parameters.
    * | Set the chemical potential with ``mu`` in ``band_params``. If
      | ``mu`` is set in ``band_params``, it is assumed the energy
      | dispersion is shifted by the chemical potential, so the
      | associated variable is set to 0.0 in the returned parameters.
    * | Set the dispersion relation with a default tight-binding model;
      | If ``dispersion`` is not given, it is automatically generated
      | from the ``band_params`` using ``get_tight_binding_dispersion``.
      | See ``get_tight_binding_dispersion`` for the list of parameters
      | and the resulting expression.
    * | Build the scattering function using predefined
      | ``scattering_models`` and the ``scattering_params``
      | associated with them.
      | See :func:`elecboltz.scatrate.build_scattering_function`
      | for supported scattering models and their parameters.
      | ``scattering_models`` is assumed to be only one ``isotropic``
      | model if not specified.
    * | Build the scattering kernel using a predefined list of
      | ``scattering_kernel_names`` and the ``scattering_kernel_params``
      | associated with each of them. See
      | :func:`elecboltz.kernel.build_kernel` for supported kernels
      | and their parameters. For spin fluctuations in particular,
      | you can specify the connecting wavevector in terms of a
      | multiple of the inverse of the unit cell by using the
      | parameter ``'Qratio'``, e.g. ``Qratio=[pi, pi, 0]``. This is
      | only possible here because the unit cell information is
      | available in the parameters.

    Parameters
    ----------
    params
        Simplified (easy-to-use) parameters for the simulation.
    
    Returns
    -------
    dict
        Parameters compatible with the classes in the package.
    """
    params = deepcopy(params)
    # load parameters from a fit
    if 'load_fit' in params:
        with open(params['load_fit'], 'r') as f:
            all_params = json.load(f)
            if 'fixed_params' in all_params:
                _deep_update(params, all_params['fixed_params'])
            if 'fit_params' in all_params:
                _deep_update(params, all_params['fit_params'])
            else: # Full scattering rate fit
                _deep_update(params, all_params['init_params'])
                params['scattering_rate'] = all_params['scattering_rate']
    new_params = deepcopy(params)
    # unit cell dimensions indicated by axis names
    if 'a' in params:
        unit_cell = [params['a'], params['a'], params['a']]
        if 'b' in params:
            unit_cell[1] = params['b']
        if 'c' in params:
            unit_cell[2] = params['c']
        new_params['unit_cell'] = unit_cell
    # some like to shift the energy dispersion itself by the chemical
    # potential and treat similarly to the other energy parameters
    if 'band_params' in params:
        new_params['band_params'] = params['band_params'].copy()
        if 'mu' in params['band_params']:
            new_params['chemical_potential'] = 0.0
    # scale all energy parameters by a given factor
    if 'energy_scale' in params:
        if not new_params.get('is_energy_scaled', False):
            for key in new_params['band_params']:
                new_params['band_params'][key] *= params['energy_scale']
            new_params['is_energy_scaled'] = True
    # get the default tight-binding dispersion relation
    if 'dispersion' not in new_params:
        new_params['dispersion'] = get_tight_binding_dispersion(
            new_params['band_params'])
    # automatically build the scattering function from named parameters
    if 'scattering_params' in params and 'scattering_rate' not in params:
        if 'scattering_models' not in params:
            new_params['scattering_models'] = ['isotropic']
        new_params['scattering_rate'] = build_scattering_function(
            new_params['scattering_params'], new_params['scattering_models'])
    if 'scattering_kernel_params' in params:
        if 'scattering_kernel_names' not in params:
            new_params['scattering_kernel_names'] = ['isotropic']
        for i, name in enumerate(new_params['scattering_kernel_names']):
            if name == 'spin_fluctuation':
                if 'Qratio' in new_params['scattering_kernel_params'][i]:
                    Qratio = new_params['scattering_kernel_params'][i]['Qratio']
                    a, b, c = new_params['unit_cell']
                    new_params['scattering_kernel_params'][i]['Q'] = [
                        Qratio[0] / a, Qratio[1] / b, Qratio[2] / c]
        new_params['scattering_kernel'] = build_kernel(
            new_params['scattering_kernel_names'],
            new_params['scattering_kernel_params'])
    return new_params


def get_tight_binding_dispersion(band_params: dict | set) -> str:
    """
    Get the tight-binding dispersion relation containing terms relating
    to the parameters in ``band_params``.

    The full tight-binding dispersion relation is given by::

        -mu - 2*t * (cos(a*kx)+cos(b*ky))
        - 4*tp * cos(a*kx)*cos(b*ky)
        - 2*tpp * (cos(2*a*kx)+cos(2*b*ky))
        - 2*tz * (cos(a*kx)-cos(b*ky))**2
            * cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)

    The list of parameters is as follows:

    * ``mu``: Chemical potential.
    * ``t``: Nearest-neighbor hopping parameter in the x-y plane.
    * ``tp``: Next-nearest-neighbor hopping parameter in the x-y plane.
    * ``tpp``: Next-next-nearest-neighbor hopping parameter
      in the x-y plane.
    * ``tz``: Nearest-neighbor hopping parameter between
      the different layers in the z direction.

    Parameters
    ----------
    band_params
        Dictionary or set of parameters for the tight-binding model.
    
    Returns
    -------
    str
        The dispersion relation expression string
    """
    dispersion = ""
    if 'mu' in band_params:
        dispersion += "-mu"
    if 't' in band_params:
        dispersion += "-2*t*(cos(a*kx)+cos(b*ky))"
    if 'tp' in band_params:
        dispersion += "-4*tp*cos(a*kx)*cos(b*ky)"
    if 'tpp' in band_params:
        dispersion += "-2*tpp*(cos(2*a*kx)+cos(2*b*ky))"
    if 'tz' in band_params:
        dispersion += "-2*tz*(cos(a*kx)-cos(b*ky))**2"
        dispersion += "*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)"
    return dispersion


def _deep_update(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst:
                dst[k] = _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    if isinstance(dst, list) and isinstance(src, list):
        for i, v in enumerate(src):
            if i < len(dst):
                dst[i] = _deep_update(dst[i], v)
            else:
                dst.append(v)
        return dst

    return src
