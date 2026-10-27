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


@pytest.mark.parametrize('model', ['constant', 'linear', 'piecewise'])
@pytest.mark.parametrize('edges', [[1.0, 2.0, 10.0], [1.0, 3.0, 7.0, 10.0]])
@pytest.mark.parametrize('verbosity', ['0', '1'])
def test_isotropic_background_averages_requested_energy_bins(tmp_path, monkeypatch, capsys, model, edges, verbosity):
    from scipy.integrate import quad
    import tdpy

    energy = np.array([1.0, 2.0, 4.0, 6.0, 10.0])  # [GeV]
    spectra = {
        'constant': np.full(5, 0.003),
        'linear': (1.0 + 0.2 * energy) * 0.001,
        'piecewise': np.array([2.0, 5.0, 1.0, 7.0, 3.0]) * 0.001,
    }  # [1/cm^2/s/sr/GeV]
    brightness = spectra[model]  # [1/cm^2/s/sr/GeV]
    directory = tmp_path / 'data'
    directory.mkdir()
    path = directory / 'iso_P8R2_SOURCE_V6_v06.txt'
    np.savetxt(path, np.column_stack((energy * 1000.0, brightness / 1000.0)))
    original_bytes = path.read_bytes()
    bins = np.array(edges)  # [GeV]
    original_bins = bins.copy()
    expected = np.array([
        quad(lambda value: np.interp(value, energy, brightness), lower, upper,
             points=energy[(energy > lower) & (energy < upper)])[0] / (upper - lower)
        for lower, upper in zip(bins[:-1], bins[1:])
    ])
    monkeypatch.setenv('TDPY_PATH', str(tmp_path))
    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    capsys.readouterr()

    with np.errstate(all='raise'):
        result = tdpy.retr_isot(bins, numbside=2)

    assert result.shape == (bins.size - 1, 48)
    np.testing.assert_allclose(result, np.broadcast_to(expected[:, None], result.shape), rtol=2e-14)
    np.testing.assert_array_equal(bins, original_bins)
    assert path.read_bytes() == original_bytes
    assert capsys.readouterr().out == (f'Reading from {path}...\n' if verbosity == '1' else '')


@pytest.mark.parametrize('channel,column', [('e', 4), ('mu', 7), ('tau', 10), ('b', 13)])
@pytest.mark.parametrize('particle,filename', [('el', 'AtProduction_positrons'), ('ph', 'AtProduction_gammas')])
@pytest.mark.parametrize('verbosity', ['0', '1'])
def test_dark_matter_table_loader_preserves_channel_grid_and_narration(
    tmp_path, monkeypatch, capsys, channel, column, particle, filename, verbosity
):
    import tdpy

    directory = tmp_path / 'data' / 'p4dm'
    directory.mkdir(parents=True)
    path = directory / f'{filename}.dat'
    masses = np.repeat([2.0, 1.0], 3)  # [GeV]
    energy_fractions = np.array([0.01, 0.1, 1.0])
    table = np.zeros((6, 14))
    table[:, 0] = masses
    table[:, 1] = np.tile(np.log10(energy_fractions), 2)
    for channel_column in (4, 7, 10, 13):
        table[:, channel_column] = channel_column + 10.0 * masses + np.tile(np.arange(3), 2)
    np.savetxt(path, table)
    original_bytes = path.read_bytes()
    monkeypatch.setenv('TDPY_PATH', str(tmp_path))
    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    capsys.readouterr()

    spectra, fractions, mass_grid = tdpy.retr_p4dm_spec(channel, part=particle)

    expected = column + 10.0 * np.array([1.0, 2.0])[None, :] + np.arange(3)[:, None]
    np.testing.assert_array_equal(spectra, expected)
    np.testing.assert_allclose(fractions, energy_fractions)
    np.testing.assert_array_equal(mass_grid, [1000.0, 2000.0])
    assert spectra.shape == (3, 2)
    assert path.read_bytes() == original_bytes
    assert capsys.readouterr().out == (f'Reading from {path}...\n' if verbosity == '1' else '')


@pytest.mark.parametrize('second_grid', [[-2.0, -0.5], [-1.0, -2.0], [-2.0, -1.0, 0.0]])
def test_dark_matter_table_loader_rejects_misaligned_mass_grids(tmp_path, monkeypatch, second_grid):
    import tdpy

    directory = tmp_path / 'data' / 'p4dm'
    directory.mkdir(parents=True)
    path = directory / 'AtProduction_positrons.dat'
    table = np.zeros((2 + len(second_grid), 14))
    table[:2, 0] = 1.0  # [GeV]
    table[2:, 0] = 2.0  # [GeV]
    table[:2, 1] = [-2.0, -1.0]
    table[2:, 1] = second_grid
    table[:, 4] = np.arange(table.shape[0]) + 1.0
    np.savetxt(path, table)
    original_bytes = path.read_bytes()
    monkeypatch.setenv('TDPY_PATH', str(tmp_path))

    with pytest.raises(ValueError, match='energy grid'):
        tdpy.retr_p4dm_spec('e')

    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize('field', ['anch', 'part'])
def test_dark_matter_table_loader_rejects_unknown_options_before_reading(tmp_path, monkeypatch, capsys, field):
    import tdpy

    directory = tmp_path / 'data' / 'p4dm'
    directory.mkdir(parents=True)
    path = directory / 'AtProduction_positrons.dat'
    table = np.zeros((2, 14))
    table[:, 0] = [1.0, 2.0]  # [GeV]
    table[:, 1] = -1.0
    np.savetxt(path, table)
    original_bytes = path.read_bytes()
    arguments = {'anch': 'e', 'part': 'el'}
    arguments[field] = 'unknown'
    monkeypatch.setenv('TDPY_PATH', str(tmp_path))
    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    capsys.readouterr()

    with pytest.raises(ValueError, match='channel|particle'):
        tdpy.retr_p4dm_spec(**arguments)

    assert capsys.readouterr().out == ''
    assert path.read_bytes() == original_bytes


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

    from tdpy.spectral import integrate_blackbody_flux_wavelength
    flux_tess = integrate_blackbody_flux_wavelength(5778.0, 0.6, 1.0)
    assert flux_tess > 0.0
    # Higher temperature emits more flux in optical band
    assert integrate_blackbody_flux_wavelength(9000.0, 0.6, 1.0) > flux_tess


def test_evaluate_spectral_model():
    from tdpy.spectral import evaluate_spectral_model, retr_spec

    class MockGdat:
        def __init__(self):
            self.numbener = 3
            self.enerpivt = 2.0
            self.factlogtenerpivt = np.log10(np.array([1.0, 2.0, 4.0]) / 2.0)
            self.indxenerpivt = 1
            self.indxener = np.array([0, 1, 2])
            class MockBctr:
                ener = np.array([1.0, 2.0, 4.0])
            self.bctrpara = MockBctr()

    gdat = MockGdat()
    flux = np.array([10.0, 20.0])

    # Power law
    spec_powr = evaluate_spectral_model(gdat, flux, sind=1.0, spectype="powr")
    assert spec_powr.shape == (3, 2)
    # Energy grid is [1.0, 2.0, 4.0], enerpivt is 2.0, sind is 1.0 -> (E/2)^(-1)
    np.testing.assert_allclose(spec_powr[1], flux)
    np.testing.assert_allclose(spec_powr[0], flux * 2.0)
    np.testing.assert_allclose(spec_powr[2], flux * 0.5)

    # Line profile (gaus)
    spec_gaus = retr_spec(gdat, flux, elin=2.0, sigm=1.0, spectype="gaus")
    assert spec_gaus.shape == (3, 2)
    assert spec_gaus[1, 0] > spec_gaus[0, 0]
