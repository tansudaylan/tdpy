import numpy as np

from tdpy.pandeia import (
    bin_depth_precision,
    configure_blackbody_source,
    summarize_saturation,
    summarize_transit_visit,
)


def test_configure_blackbody_source_sets_spectrum_and_normalization():
    calculation = {"scene": [{"spectrum": {}}]}
    configure_blackbody_source(calculation, 5000.0, 10.0, 1600.0, 1.25)
    spectrum = calculation["scene"][0]["spectrum"]
    assert spectrum["sed"] == {"sed_type": "blackbody", "temp": 5000.0}
    assert spectrum["normalization"]["norm_wave"] == 1.25
    assert spectrum["normalization"]["norm_flux"] == 160.0


def test_summarize_saturation_counts_each_flag():
    report = {
        "2d": {"saturation_unrotated": np.array([[0, 1], [1, 2]])},
        "scalar": {"fraction_saturation": 0.8},
    }
    assert summarize_saturation(report, 3) == {
        "number_groups": 3,
        "fraction_of_etc_signal_limit": 0.8,
        "partially_saturated_pixels": 2,
        "fully_saturated_pixels": 1,
    }


def test_bin_depth_precision_preserves_report_bin_equations():
    wavelength = np.array([2.91, 2.96, 5.12])  # [micrometer]
    report = {
        "1d": {
            "extracted_flux": [wavelength, np.array([10.0, 10.0, 20.0])],
            "extracted_noise": [wavelength, np.array([1.0, 1.0, 2.0])],
        },
        "scalar": {"total_exposure_time": 10.0},
    }
    integrations, centers, single, precision = bin_depth_precision(
        report, 1.0, 0.1, (2.87, 5.18)
    )
    assert integrations == 360
    assert centers.size == single.size == precision.size
    assert single[0] == np.sqrt(2.0) / 20.0 * 1.0e6
    np.testing.assert_allclose(
        precision[0], np.sqrt(2.0) * single[0] / np.sqrt(integrations)
    )


def test_summarize_transit_visit_uses_equal_in_and_out_phases():
    wavelength = np.array([2.91, 2.96, 5.12])  # [micrometer]
    report = {
        "1d": {
            "extracted_flux": [wavelength, np.array([10.0, 10.0, 20.0])],
            "extracted_noise": [wavelength, np.array([1.0, 1.0, 2.0])],
        },
        "scalar": {"total_exposure_time": 10.0},
    }
    summary = summarize_transit_visit(report, 1.0, 0.1, (2.87, 5.18))
    assert summary["number_integrations_per_phase"] == 360
    assert summary["science_integrations"] == 720
    assert summary["science_exposure_hours"] == 2.0
    assert summary["median_depth_precision_ppm"] == np.nanmedian(
        summary["depth_precision_ppm"]
    )