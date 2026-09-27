import numpy as np

import tdpy


def test_fit_fourier_series_recovers_known_harmonics():
    """The fitted basis preserves constant, cosine, and sine ordering."""
    sample_locations = np.linspace(0., 4., 17, endpoint=False)
    phase = 2. * np.pi * sample_locations / 4.
    values = 3. + 2. * np.cos(phase) - 0.5 * np.sin(2. * phase)

    coefficients, fitted_values = tdpy.fit_fourier_series(
        values, sample_locations, period=4., harmonic_count=2
    )

    np.testing.assert_allclose(coefficients, [3., 2., 0., 0., -0.5], atol=1e-12)
    np.testing.assert_allclose(fitted_values, values, atol=1e-12)