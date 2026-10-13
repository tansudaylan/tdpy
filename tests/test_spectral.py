import numpy as np
import pytest

from tdpy.spectral import (
    blackbody_flux_energy,
    blackbody_flux_wavelength,
    blackbody_spectral_flux,
    evaluate_continuum_spectrum,
    evaluate_line_profile,
    retr_fluxspecbbod,
)


@pytest.mark.parametrize(
    ("profile", "parameters", "tolerance"),
    (
        ("gaus", {"gaussian_width": [1.0]}, 1e-6),
        ("lore", {"lorentz_width": [1.0]}, 2e-2),
        ("voig", {"gaussian_width": [1.0], "lorentz_width": [0.5]}, 2e-2),
        (
            "pvoi",
            {"gaussian_width": [1.0], "lorentz_width": [1.0], "mixing_fraction": [0.3]},
            2e-2,
        ),
        ("sinc", {"gaussian_width": [1.0]}, 3e-3),
        ("skew", {"gaussian_width": [1.0], "skewness": [4.0]}, 1e-6),
        ("toph", {"gaussian_width": [1.0]}, 2e-3),
    ),
)
def test_line_profiles_preserve_integrated_flux(profile, parameters, tolerance):
    axis = np.linspace(-50.0, 50.0, 100_001)
    values = evaluate_line_profile(axis, profile, [2.0], [0.0], **parameters)[:, 0]
    assert np.trapezoid(values, axis) == pytest.approx(2.0, rel=tolerance)


def test_unknown_line_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported line profile"):
        evaluate_line_profile([0.0, 1.0], "unknown", [1.0], [0.0])


@pytest.mark.parametrize("profile", ("powr", "curv", "expc"))
def test_continuum_spectra_use_pivoted_parameters(profile):
    energy = np.array([0.25, 0.5, 1.0])
    flux = np.array([2.0, 3.0])
    spectral_index = np.array([1.0, 2.0])
    arguments = {
        "powr": {},
        "curv": {"curvature": np.array([0.1, 0.2])},
        "expc": {"cutoff_energy": np.array([0.8, 1.2])},
    }[profile]

    spectrum = evaluate_continuum_spectrum(
        energy,
        flux,
        profile,
        0.5,
        spectral_index,
        log_energy_ratio=np.log(energy / 0.5),
        **arguments,
    )

    assert spectrum.shape == (energy.size, flux.size)
    assert np.isfinite(spectrum).all()
    expected_at_pivot = (
        flux
        if profile != "curv"
        else flux * 0.5 ** (-spectral_index)
    )
    np.testing.assert_allclose(spectrum[1], expected_at_pivot)


def test_blackbody_flux_wavelength_and_energy_are_positive_and_peak():
    flux_wavelength = blackbody_flux_wavelength(5778.0, 0.5)
    assert flux_wavelength > 0.0

    flux_energy = blackbody_flux_energy(5778.0, 2.5)
    assert flux_energy > 0.0

    # general wrapper with quantity dispatch
    flux_from_spec = blackbody_spectral_flux(5778.0, 0.5, quantity="wlen")
    assert flux_from_spec == pytest.approx(flux_wavelength)
    assert retr_fluxspecbbod(5778.0, 0.5, quantity="wlen") == pytest.approx(flux_wavelength)

    flux_band, energies = blackbody_spectral_flux(3500.0, band="AXIS0224")
    assert energies.size == 400
    assert flux_band[0] > 0.0
    assert np.any(flux_band > 0.0)


def test_band_energy_grid_and_bolometric_luminosity():
    from tdpy.spectral import (
        band_energy_grid,
        retr_blimener,
        bolometric_blackbody_luminosity,
        retr_fluxbolobbod,
        instrument_transmission,
        retr_functran,
    )

    axis_energies = band_energy_grid("AXIS0224")
    assert axis_energies.size == 400
    assert retr_blimener("AXIS0224").size == 400

    bolo_energies = band_energy_grid("bolo")
    assert bolo_energies.size == 1000

    # Solar bolometric luminosity ~ 3.826e33 erg/s
    lumi_sun = bolometric_blackbody_luminosity(5772.0, 1.0)
    assert lumi_sun == pytest.approx(3.826e33, rel=1e-2)

    # Unit pair compatibility: [temp, 'K'] and [radius, 'R_odot']
    lumi_pair = retr_fluxbolobbod([5772.0, "K"], [1.0, "R_odot"])
    assert lumi_pair[1] == "erg/s"
    assert lumi_pair[0] == pytest.approx(3.826e33, rel=1e-2)

    trans = instrument_transmission("AXIS", np.array([100.0, 500.0, 15000.0]))
    np.testing.assert_array_equal(trans, [0.0, 1.0, 0.0])
    assert retr_functran("AXIS", 1000.0) == 1.0
