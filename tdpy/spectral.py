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
    spectral_index = np.atleast_1d(np.asarray(spectral_index, dtype=float))
    if profile == "powr":
        return flux[None, :] * (energy / reference_energy)[:, None] ** (
            -spectral_index[None, :]
        )
    if profile == "curv":
        if log_energy_ratio is None:
            log_energy_ratio = np.log(energy / reference_energy)
        log_energy_ratio = np.asarray(log_energy_ratio, dtype=float).reshape(-1)
        curvature = np.atleast_1d(np.asarray(curvature, dtype=float))
        return flux[None, :] * energy[:, None] ** (
            -spectral_index[None, :]
            - log_energy_ratio[:, None] * curvature[None, :]
        )
    if profile == "expc":
        cutoff_energy = np.atleast_1d(np.asarray(cutoff_energy, dtype=float))
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
    exponent = planck * speed_of_light / (wavelength_m * boltzmann * temperature)
    safe_exponent = np.minimum(exponent, 700.0)
    inv_expm1 = np.where(exponent > 700.0, 0.0, 1.0 / np.expm1(safe_exponent))
    flux = (
        2.0
        * np.pi
        * planck
        * speed_of_light**2
        / wavelength_m**5
        * inv_expm1
    )  # [W / m^2 / m]
    return flux * 1e-3  # [erg / s / cm^2 / micron]


def integrate_blackbody_flux_wavelength(
    temperature: float | np.ndarray,
    wavelength_min_microns: float,
    wavelength_max_microns: float,
    number_wavelengths: int = 50,
) -> float | np.ndarray:
    """Integrate Planck blackbody spectral surface flux over a wavelength range in microns.

    Parameters
    ----------
    temperature : float or np.ndarray
        Blackbody temperature in Kelvin.
    wavelength_min_microns : float
        Lower wavelength bound in microns.
    wavelength_max_microns : float
        Upper wavelength bound in microns.
    number_wavelengths : int, optional
        Number of grid evaluation points (default 50).

    Returns
    -------
    float or np.ndarray
        Integrated surface flux in erg / s / cm^2.
    """
    wavelengths = np.linspace(wavelength_min_microns, wavelength_max_microns, number_wavelengths)
    flux = blackbody_flux_wavelength(temperature, wavelengths)
    integrated = np.trapz(flux, wavelengths, axis=-1 if np.ndim(flux) > 1 else 0)
    if np.ndim(temperature) == 0:
        return float(integrated)
    return integrated


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
    exponent = energy_joule / (boltzmann * temperature)
    safe_exponent = np.minimum(exponent, 700.0)
    inv_expm1 = np.where(exponent > 700.0, 0.0, 1.0 / np.expm1(safe_exponent))
    flux = (
        2.0
        * np.pi
        * energy_joule**3
        / planck**3
        / speed_of_light**2
        * inv_expm1
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
            energies = band_energy_grid(band)
            return blackbody_flux_energy(temperature, energies), energies
        raise ValueError(f"Unknown spectral band {band!r}.")

    if quantity == "evol":
        return blackbody_flux_energy(temperature, values)
    if quantity in ("wlen", "wavelength"):
        return blackbody_flux_wavelength(temperature, values)
    raise ValueError(f"quantity must be 'evol', 'wlen', or 'wavelength', got {quantity!r}.")


def band_energy_grid(band: str, *, typeband: str | None = None) -> np.ndarray:
    """Return an energy grid in electronvolts (eV) for a supported spectral band."""
    if band is None and typeband is not None:
        band = typeband
    if band == "AXIS0224":
        return np.logspace(np.log10(200.0), np.log10(2400.0), 400)  # [eV]
    if band == "bolo":
        return np.logspace(-5.0, 6.0, 1000)  # [eV]
    raise ValueError(f"Unknown spectral band {band!r}.")


retr_blimener = band_energy_grid


def bolometric_blackbody_luminosity(
    temperature_kelvin: float | np.ndarray | tuple | list,
    radius_solar: float | np.ndarray | tuple | list,
) -> float | np.ndarray | list:
    """Calculate bolometric blackbody luminosity in erg/s."""
    is_pair = False
    if isinstance(temperature_kelvin, (list, tuple)) and len(temperature_kelvin) == 2:
        t_val = temperature_kelvin[0]
        r_val = radius_solar[0]
        is_pair = True
    else:
        t_val = temperature_kelvin
        r_val = radius_solar

    # 4 * pi * R_sun^2 * sigma_sb = 3.447e18 erg / (s K^4 R_sun^2)
    stefan_boltzmann_solar_cgs = 3.447e18  # [erg / (s K^4 R_sun^2)]
    lumi = stefan_boltzmann_solar_cgs * np.asarray(t_val, dtype=float)**4 * np.asarray(r_val, dtype=float)**2
    if is_pair:
        return [float(lumi) if np.ndim(lumi) == 0 else lumi, "erg/s"]
    return lumi


retr_fluxbolobbod = bolometric_blackbody_luminosity


def instrument_transmission(
    instrument: str | None = None,
    energy_ev: float | np.ndarray | None = None,
    *,
    typeinst: str | None = None,
    xdat: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return transmission efficiency across energy for a given instrument."""
    if instrument is None and typeinst is not None:
        instrument = typeinst
    if energy_ev is None and xdat is not None:
        energy_ev = xdat
    if energy_ev is None:
        if instrument == "AXIS":
            energy_ev = np.linspace(200.0, 10000.0, 1000)
        else:
            raise ValueError(f"Energy grid required for instrument {instrument!r}.")

    energy = np.asarray(energy_ev, dtype=float)
    if instrument == "AXIS":
        transmission = np.zeros_like(energy)
        mask = (energy >= 200.0) & (energy <= 10000.0)
        transmission[mask] = 1.0
        return transmission
    raise ValueError(f"Unknown instrument {instrument!r}.")


retr_functran = instrument_transmission


def evaluate_spectral_model(
    gdat,
    flux,
    sind=None,
    curv=None,
    expc=None,
    sindcolr=None,
    elin=None,
    edisintp=None,
    sigm=None,
    gamm=None,
    frac=None,
    skew=None,
    spectype="powr",
    plot=False,
):
    """Evaluate line, color, or continuum spectra across the energy grid of the experiment state."""
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    if getattr(gdat, "numbener", None) == 1:
        return flux[None, :]

    meanener = np.asarray(
        getattr(gdat.bctrpara, "enerplot", gdat.bctrpara.ener) if plot else gdat.bctrpara.ener,
        dtype=float,
    )

    if spectype in ("gaus", "voig", "edis", "pvoi", "lore", "sinc", "skew", "toph"):
        if elin is None:
            raise ValueError("elin is required for line spectra.")
        elin = np.atleast_1d(np.asarray(elin, dtype=float))
        if elin.size == 1 and flux.size > 1:
            elin = np.full(flux.size, elin.item())
        if elin.size != flux.size:
            raise ValueError("elin must contain one center per flux value.")

        def _line_param(val, name, require_pos=True):
            if val is None:
                return None
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            if arr.size == 1 and flux.size > 1:
                arr = np.full(flux.size, arr.item())
            if arr.size != flux.size or not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain finite values, one per line.")
            if require_pos and np.any(arr <= 0.0):
                raise ValueError(f"{name} must contain positive values, one per line.")
            return arr

        if spectype == "edis":
            if edisintp is None:
                raise ValueError("edisintp is required for spectype='edis'.")
            gaussian_width = _line_param(edisintp(elin), "edisintp(elin)")
        elif spectype in ("gaus", "voig", "pvoi"):
            if sigm is None:
                raise ValueError("sigm is required for Gaussian-based lines.")
            gaussian_width = _line_param(sigm, "sigm")
        else:
            gaussian_width = _line_param(sigm, "sigm") if sigm is not None else None

        if spectype in ("lore", "voig", "pvoi"):
            if gamm is None:
                raise ValueError("gamm is required for Lorentzian-based lines.")
            lorentz_width = _line_param(gamm, "gamm")
        else:
            lorentz_width = _line_param(gamm, "gamm") if gamm is not None else None

        mixing_fraction = None
        if spectype == "pvoi":
            if frac is None:
                raise ValueError("frac is required for spectype='pvoi'.")
            mixing_fraction = _line_param(frac, "frac", require_pos=False)
            if np.any((mixing_fraction < 0.0) | (mixing_fraction > 1.0)):
                raise ValueError("frac must contain values between zero and one, one per line.")

        skewness = _line_param(skew, "skew", require_pos=False) if skew is not None else None

        return evaluate_line_profile(
            meanener,
            spectype,
            flux,
            elin,
            gaussian_width=gaussian_width,
            lorentz_width=lorentz_width,
            mixing_fraction=mixing_fraction,
            skewness=skewness,
        )

    if spectype == "colr":
        if plot:
            return np.zeros((meanener.size, flux.size))
        if sindcolr is None:
            raise ValueError("sindcolr is required for spectype='colr'.")
        spec = np.zeros((meanener.size, flux.size))
        indxener = np.arange(meanener.size) if plot else gdat.indxener
        for i in indxener:
            if i == gdat.indxenerpivt:
                spec[i] = flux
            else:
                indxcolr = i if i < gdat.indxenerpivt else i - 1
                if indxcolr >= len(sindcolr):
                    indxcolr = len(sindcolr) - 1
                spec[i] = flux * (meanener[i] / gdat.enerpivt) ** (-sindcolr[indxcolr])
        return spec

    if spectype in ("powr", "curv", "expc"):
        return evaluate_continuum_spectrum(
            meanener,
            flux,
            spectype,
            gdat.enerpivt,
            sind,
            curvature=curv,
            cutoff_energy=expc,
            log_energy_ratio=getattr(gdat, "factlogtenerpivt", None),
        )

    raise ValueError(f"Unsupported spectype {spectype!r}.")


retr_spec = evaluate_spectral_model
retr_fluxspecbbodwlen = blackbody_flux_wavelength
retr_fluxspecbbodener = blackbody_flux_energy
retr_fluxspecbbod = blackbody_spectral_flux

__all__ = [
    "band_energy_grid",
    "blackbody_flux_energy",
    "blackbody_flux_wavelength",
    "blackbody_spectral_flux",
    "bolometric_blackbody_luminosity",
    "evaluate_continuum_spectrum",
    "evaluate_line_profile",
    "evaluate_spectral_model",
    "instrument_transmission",
    "integrate_blackbody_flux_wavelength",
    "retr_blimener",
    "retr_fluxbolobbod",
    "retr_fluxspecbbod",
    "retr_fluxspecbbodener",
    "retr_fluxspecbbodwlen",
    "retr_functran",
    "retr_spec",
]
