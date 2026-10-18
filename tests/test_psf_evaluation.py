import numpy as np
import pytest
from scipy.integrate import trapezoid

from tdpy import evaluate_psf_profile


@pytest.mark.parametrize("model,parameters", [
    ("singgaus", None),
    ("singking", [0.3, 2.5]),
    ("doubking", [0.3, 2.5, 0.8, 3.0, 0.7]),
])
def test_explicit_psf_parameters_and_spherical_grid_normalization(model, parameters, monkeypatch):
    monkeypatch.delattr(np, 'trapz', raising=False)
    angles = np.linspace(0.0, 1.0, 501)  # [rad]
    psfp = None if parameters is None else np.asarray(parameters, dtype=float)[:, None, None]

    profile = evaluate_psf_profile(
        {"sigc": 0.3}, np.array([0]), angles, model, typenormangl="ferm",
        psfp=psfp, indxpsfpinit=0 if psfp is not None else None,
        fermscalfact=np.ones((1, 1)),
    )

    assert profile.shape == (1, angles.size, 1)
    assert np.isfinite(profile).all()
    integral = 2.0 * np.pi * trapezoid(profile[0, :, 0] * np.sin(angles), angles)
    assert integral == pytest.approx(1.0)


def test_king_profile_requires_parameter_array_and_start_index():
    with pytest.raises(ValueError, match="psfp and indxpsfpinit"):
        evaluate_psf_profile({}, np.array([0]), np.linspace(0.0, 1.0, 51), "singking")


def test_legacy_tdpy_psf_entry_point_remains_distinct():
    import tdpy

    assert tdpy.retr_psfn is not evaluate_psf_profile
