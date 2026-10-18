"""Point-spread profiles with explicit energy and event-class parameters."""

import numpy as np
from scipy.integrate import trapezoid


def evaluate_psf_profile(
    dictpara,
    indxenertemp,
    arryangl,
    typemodlpsfn,
    typenormangl='none',
    booldiag=True,
    *,
    psfp=None,
    indxpsfpinit=None,
    fermscalfact=None,
):
    """Evaluate Gaussian or King-family profiles, optionally normalized over an angular grid."""

    from .util import retr_singgaus, retr_singking, retr_doubking

    angle = np.asarray(arryangl, dtype=float)
    if angle.ndim != 1:
        raise ValueError('arryangl must be one-dimensional.')
    if typenormangl not in ('none', 'ferm'):
        raise ValueError("typenormangl must be 'none' or 'ferm'.")
    if typenormangl == 'ferm':
        if fermscalfact is None:
            raise ValueError("fermscalfact is required for typenormangl='ferm'.")
        scale = np.asarray(fermscalfact, dtype=float)
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError('fermscalfact must contain finite positive values.')
        angle = 2.0 * np.arcsin(np.sqrt(2.0 - 2.0 * np.cos(angle)) / 2.0)
        angle_scale = angle[None, :, None] / scale[:, None, :]
    else:
        angle_scale = angle[None, :, None]

    def shape_parameters(number):
        if psfp is None or indxpsfpinit is None:
            raise ValueError('psfp and indxpsfpinit are required for King PSFs.')
        values = np.asarray(psfp[indxpsfpinit:indxpsfpinit + number], dtype=float)
        if values.ndim != 3 or values.shape[0] != number:
            raise ValueError('psfp must have shape (parameters, energy, event class).')
        return [values[index, :, None, :] for index in range(number)]

    if typemodlpsfn == 'singgaus':
        if 'sigc' not in dictpara:
            raise ValueError("dictpara['sigc'] is required for a Gaussian PSF.")
        sigma = np.asarray(dictpara['sigc'], dtype=float)
        if booldiag and (not np.isfinite(sigma).all() or np.any(sigma == 0.0)):
            raise ValueError('Gaussian PSF width must be finite and nonzero.')
        profile = retr_singgaus(angle_scale, sigma)
    elif typemodlpsfn == 'singking':
        sigma_core, gamma_core = shape_parameters(2)
        profile = retr_singking(angle_scale, sigma_core, gamma_core)
    elif typemodlpsfn == 'doubking':
        sigma_core, gamma_core, sigma_tail, gamma_tail, fraction = shape_parameters(5)
        profile = retr_doubking(angle_scale, fraction, sigma_core, gamma_core, sigma_tail, gamma_tail)
    else:
        raise ValueError('Unsupported PSF model %r.' % typemodlpsfn)
    if typenormangl == 'ferm':
        normalization = 2.0 * np.pi * trapezoid(
            profile * np.sin(angle[None, :, None]), angle, axis=1,
        )[:, None, :]
        profile = profile / normalization
    return profile