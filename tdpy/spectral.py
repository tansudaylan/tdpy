"""Normalized analytic line profiles shared by spectral-model consumers."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, wofz


def evaluate_line_profile(
    axis,
    profile: str,
    flux,
    center,
    gaussian_width=None,
    lorentz_width=None,
    mixing_fraction=None,
    skewness=None,
):
    """Return integrated-flux-normalized line profiles on a one-dimensional axis."""

    axis = np.asarray(axis, dtype=float).reshape(-1, 1)
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    center = np.atleast_1d(np.asarray(center, dtype=float))
    if axis.size == 0 or flux.size != center.size or not np.isfinite(axis).all():
        raise ValueError("axis must be finite and flux/center arrays must have equal nonzero length")
    if not np.isfinite(flux).all() or not np.isfinite(center).all():
        raise ValueError("line fluxes and centers must be finite")
    offset = axis - center[None, :]

    def positive(values, name):
        values = np.atleast_1d(np.asarray(values, dtype=float))
        if values.size != flux.size or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{name} must contain one finite positive value per line")
        return values

    if profile in ("gaus", "voig", "pvoi", "sinc", "skew", "toph", "edis"):
        gaussian_width = positive(gaussian_width, "gaussian_width")
    if profile in ("lore", "voig", "pvoi"):
        lorentz_width = positive(lorentz_width, "lorentz_width")

    if profile in ("gaus", "edis"):
        shape = np.exp(-0.5 * (offset / gaussian_width) ** 2) / (
            gaussian_width * np.sqrt(2.0 * np.pi)
        )
    elif profile == "lore":
        shape = lorentz_width / (np.pi * (offset**2 + lorentz_width**2))
    elif profile == "voig":
        argument = (offset + 1j * lorentz_width) / (
            np.sqrt(2.0) * gaussian_width
        )
        shape = np.real(wofz(argument)) / (gaussian_width * np.sqrt(2.0 * np.pi))
    elif profile == "pvoi":
        fraction = np.atleast_1d(np.asarray(mixing_fraction, dtype=float))
        if fraction.size != flux.size or np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError("mixing_fraction must lie between zero and one for every line")
        gaussian = np.exp(-0.5 * (offset / gaussian_width) ** 2) / (
            gaussian_width * np.sqrt(2.0 * np.pi)
        )
        lorentzian = lorentz_width / (np.pi * (offset**2 + lorentz_width**2))
        shape = (1.0 - fraction) * gaussian + fraction * lorentzian
    elif profile == "sinc":
        shape = np.sinc(offset / gaussian_width) ** 2 / gaussian_width
    elif profile == "skew":
        skewness = np.atleast_1d(np.asarray(skewness, dtype=float))
        if skewness.size != flux.size or not np.isfinite(skewness).all():
            raise ValueError("skewness must contain one finite value per line")
        standardized = offset / gaussian_width
        gaussian = np.exp(-0.5 * standardized**2) / (
            gaussian_width * np.sqrt(2.0 * np.pi)
        )
        shape = 2.0 * gaussian * ndtr(skewness * standardized)
    elif profile == "toph":
        shape = (np.abs(offset) <= 0.5 * gaussian_width) / gaussian_width
    else:
        raise ValueError(f"Unsupported line profile {profile!r}.")
    return flux[None, :] * shape


def evaluate_continuum_spectrum(
    energy,
    flux,
    profile: str,
    reference_energy,
    spectral_index,
    curvature=None,
    cutoff_energy=None,
    log_energy_ratio=None,
):
    """Evaluate shared power-law, curved, and exponential-cutoff continua."""

    energy = np.asarray(energy, dtype=float).reshape(-1)
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    spectral_index = np.asarray(spectral_index, dtype=float)
    if profile == "powr":
        return flux[None, :] * (energy / reference_energy)[:, None] ** (
            -spectral_index[None, :]
        )
    if profile == "curv":
        if log_energy_ratio is None:
            log_energy_ratio = np.log(energy / reference_energy)
        log_energy_ratio = np.asarray(log_energy_ratio, dtype=float).reshape(-1)
        curvature = np.asarray(curvature, dtype=float)
        return flux[None, :] * energy[:, None] ** (
            -spectral_index[None, :]
            - log_energy_ratio[:, None] * curvature[None, :]
        )
    if profile == "expc":
        cutoff_energy = np.asarray(cutoff_energy, dtype=float)
        return (
            flux[None, :]
            * (energy / reference_energy)[:, None] ** (-spectral_index[None, :])
            * np.exp(-(energy - reference_energy)[:, None] / cutoff_energy[None, :])
        )
    raise ValueError(f"Unsupported continuum profile {profile!r}.")


def blackbody_flux_wavelength(
    temperature: float | np.ndarray,
    wavelength_microns: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate Planck blackbody spectral surface flux in erg / s / cm^2 / micron."""

    planck = 6.62607015e-34  # [J s]
    speed_of_light = 299792458.0  # [m / s]
    boltzmann = 1.380649e-23  # [J / K]
    wavelength_m = np.asarray(wavelength_microns, dtype=float) * 1e-6  # [m]
    temperature = np.asarray(temperature, dtype=float)  # [K]
    flux = (
        2.0
        * np.pi
        * planck
        * speed_of_light**2
        / wavelength_m**5
        / np.expm1(planck * speed_of_light / (wavelength_m * boltzmann * temperature))
    )  # [W / m^2 / m]
    return flux * 1e-3  # [erg / s / cm^2 / micron]


def blackbody_flux_energy(
    temperature: float | np.ndarray,
    energy_ev: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate Planck blackbody spectral surface flux in erg / s / cm^2 / eV."""

    planck = 6.62607015e-34  # [J s]
    speed_of_light = 299792458.0  # [m / s]
    boltzmann = 1.380649e-23  # [J / K]
    joules_per_ev = 1.602176634e-19  # [J / eV]
    energy_joule = np.asarray(energy_ev, dtype=float) * joules_per_ev  # [J]
    temperature = np.asarray(temperature, dtype=float)  # [K]
    flux = (
        2.0
        * np.pi
        * energy_joule**3
        / planck**3
        / speed_of_light**2
        / np.expm1(energy_joule / (boltzmann * temperature))
    )
    return flux * 1.602176634e-16  # [erg / s / cm^2 / eV]


def blackbody_spectral_flux(
    temperature: float | np.ndarray,
    values: float | np.ndarray | None = None,
    *,
    xdat: float | np.ndarray | None = None,
    band: str | None = None,
    typeband: str | None = None,
    quantity: str = "wavelength",
    typexdat: str | None = None,
) -> float | np.ndarray | tuple[float | np.ndarray, np.ndarray]:
    """Calculate blackbody spectral surface flux for wavelength or energy."""

    if values is None and xdat is not None:
        values = xdat
    if band is None and typeband is not None:
        band = typeband
    if typexdat is not None:
        quantity = typexdat

    if values is None and band is None:
        raise ValueError("Either values or band must be provided.")
    if band is not None:
        if band == "AXIS0224":
            energies = np.logspace(np.log10(200.0), np.log10(2400.0), 400)  # [eV]
            return blackbody_flux_energy(temperature, energies), energies
        raise ValueError(f"Unknown spectral band {band!r}.")

    if quantity == "evol":
        return blackbody_flux_energy(temperature, values)
    if quantity in ("wlen", "wavelength"):
        return blackbody_flux_wavelength(temperature, values)
    raise ValueError(f"quantity must be 'evol', 'wlen', or 'wavelength', got {quantity!r}.")


retr_fluxspecbbodwlen = blackbody_flux_wavelength
retr_fluxspecbbodener = blackbody_flux_energy
retr_fluxspecbbod = blackbody_spectral_flux

__all__ = [
    "blackbody_flux_energy",
    "blackbody_flux_wavelength",
    "blackbody_spectral_flux",
    "evaluate_continuum_spectrum",
    "evaluate_line_profile",
    "retr_fluxspecbbod",
    "retr_fluxspecbbodener",
    "retr_fluxspecbbodwlen",
]