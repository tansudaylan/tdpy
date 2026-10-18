import matplotlib.image as mpimg
import numpy as np
import pytest
from scipy.integrate import trapezoid

from tdpy.spectral import blackbody_spectral_flux, retr_dictfluxband, retr_dictxraystar


def test_band_flux_uses_canonical_blackbody_and_reusable_transmission(monkeypatch):
    monkeypatch.delattr(np, 'trapz', raising=False)
    fluxes, state = retr_dictfluxband(6000.0, ['TESS', 'ULTRASAT', 'Bolometric'])
    spectrum = blackbody_spectral_flux(6000.0, xdat=state.midpwlen, typexdat='wlen')

    for band in fluxes:
        assert fluxes[band] == pytest.approx(trapezoid(spectrum * state.dictfunctran[band], x=state.midpwlen))
    assert fluxes['Bolometric'] > fluxes['TESS'] > 0
    repeated, reused = retr_dictfluxband(6000.0, ['TESS', 'ULTRASAT', 'Bolometric'], gdatfluxband=state)
    assert reused is state
    assert repeated == fluxes


def test_band_flux_optional_plot_is_valid(tmp_path):
    retr_dictfluxband(6000.0, ['TESS'], pathvisutarg=tmp_path, strgtarg='toy')

    image = mpimg.imread(tmp_path / 'fluxband_toy_TESS.png')
    assert image[..., :3].min() < 0.8


def test_xray_prediction_retains_assumed_fraction_and_distance_scaling():
    temperatures = [np.array([3500.0, 6000.0]), 'K']
    radii = [np.array([0.5, 1.0]), 'solar radii']
    first = retr_dictxraystar(temperatures, radii, [np.array([10.0, 10.0]), 'pc'])
    second = retr_dictxraystar(temperatures, radii, [np.array([20.0, 20.0]), 'pc'])

    np.testing.assert_allclose(first['fracxrayboloaddi'][0], [1e-4, 6e-7])
    np.testing.assert_allclose(first['fluxpred0224'][0], 4.0 * second['fluxpred0224'][0])
    assert np.isnan(retr_dictxraystar(temperatures, radii, None)['fluxpred0224'][0]).all()
