import numpy as np
import pytest

from tdpy.exoplanet import (
    empirical_interval_contrasts,
    fold_tser,
    keplerian_radial_velocity,
    kipping_to_quadratic_limb_darkening,
    paired_interval_contrast_forecast,
    plot_exoplanet_discovery_timeline,
    plot_exoplanet_period_radius,
    plot_atmosphere_spectra,
    plot_paired_band_contrasts,
    quadratic_limb_darkened_stellar_grid,
    quadratic_limb_darkening,
    quadratic_to_kipping_limb_darkening,
    keplerian_orbit_radial_velocity,
    keplerian_semi_amplitude,
    relative_flux_from_delta_magnitude,
    retr_albg,
    retr_alphelli,
    retr_brgtlmdk,
    retr_deptbeam,
    retr_deptelli,
    retr_duratranfull,
    retr_duratrantotl,
    retr_esmm,
    retr_imfa,
    retr_lcurmodl_flarsing,
    retr_listcolrcomp,
    retr_liststrgcomp,
    retr_logg,
    retr_perikepl,
    retr_radifrommass,
    retr_radihill,
    retr_radiroch,
    retr_reso,
    retr_rflxfromdmag,
    retr_rs2a,
    retr_rsmacomp,
    retr_rvel,
    retr_rvelsema,
    retr_scalheig,
    retr_smaxkepl,
    retr_tmptplandayynigh,
    retr_tsmm,
    retr_vesc,
    stellar_limb_darkened_brightness,
)


def test_keplerian_radial_velocity_reduces_to_a_sinusoid_for_circular_orbits():
    time = np.linspace(0.0, 30.0, 200)  # [day]
    velocity = keplerian_radial_velocity(time, 7.0, 3.0, 0.0, 0.4, 1.1)
    expected = 3.0 * np.cos(0.4 + 1.1 + 2.0 * np.pi * time / 7.0)
    assert velocity == pytest.approx(expected, abs=1e-10)


def test_keplerian_radial_velocity_has_zero_time_average_and_broadcasts():
    time = np.linspace(0.0, 10.0, 20001)[:-1]  # [day]
    eccentricity = np.array([0.0, 0.3, 0.7])
    velocity = keplerian_radial_velocity(time[:, None], 10.0, 5.0, eccentricity, 1.0, 0.2)
    assert velocity.shape == (time.size, 3)
    assert np.mean(velocity, axis=0) == pytest.approx(np.zeros(3), abs=1e-6)
    assert np.ptp(velocity, axis=0) == pytest.approx(
        2.0 * 5.0 * np.ones(3), rel=1e-3
    )


def test_kipping_limb_darkening_transforms_round_trip():
    coefficients = np.array((0.4, 0.25))
    kipping_parameters = quadratic_to_kipping_limb_darkening(*coefficients)

    assert kipping_parameters == pytest.approx((0.4225, 0.3076923077))
    assert kipping_to_quadratic_limb_darkening(*kipping_parameters) == pytest.approx(coefficients)


def test_quadratic_limb_darkening_and_grid_preserve_disk_geometry():
    intensity = quadratic_limb_darkening(np.array([0.0, 1.0]), (0.4, 0.25))
    image_x, image_y, radial_distance, brightness = quadratic_limb_darkened_stellar_grid(
        5, (0.4, 0.25), image_limit=2.0
    )

    assert intensity == pytest.approx((0.35, 1.0))
    assert image_x.shape == image_y.shape == radial_distance.shape == brightness.shape == (5, 5)
    assert brightness[2, 2] == pytest.approx(1.0)
    assert brightness[2, 3] == pytest.approx(0.35)
    assert brightness[0, 0] == 0.0


def test_interval_contrast_forecasts_preserve_paired_differences():
    centers = np.array([1.0, 1.1, 1.2, 1.3])
    definitions = {"feature": {"feature": (1.0, 1.2), "continuum": ((1.2, 1.4),)}}
    empirical = empirical_interval_contrasts(
        centers, np.array([2.0, 2.0, 0.0, 0.0]), np.eye(4), definitions
    )
    assert empirical["contrasts"]["feature"]["injected_contrast_ppm"] == 2.0
    paired = paired_interval_contrast_forecast(
        {"d": centers, "e": centers},
        {
            "d": {"model": [2.0, 2.0, 0.0, 0.0]},
            "e": {"model": [1.0, 1.0, 0.0, 0.0]},
        },
        {"d": np.eye(4), "e": np.eye(4)},
        definitions,
    )
    assert paired["models"]["model"]["d_minus_e_contrast_ppm"] == [1.0]


def test_exoplanet_forecast_plots_write_requested_files(tmp_path):
    bands = {"feature": {"label": "X", "feature": (1.0, 1.2)}}
    centers = {"b": np.array([1.0, 1.1]), "c": np.array([1.0, 1.1])}
    spectra = {name: {"model": np.array([-1.0, 1.0])} for name in centers}
    atmosphere_path = plot_atmosphere_spectra(
        centers,
        spectra,
        {name: np.zeros(2) for name in centers},
        {name: np.ones(2) for name in centers},
        bands,
        (("model", "-"),),
        tmp_path / "atmosphere.png",
    )
    contrast_path = plot_paired_band_contrasts(
        {
            "difference_uncertainty_ppm": [1.0],
            "models": {"model": {"d_minus_e_contrast_ppm": [2.0]}},
        },
        bands,
        (("model", "model", "black", "o"),),
        tmp_path / "contrast.pdf",
    )
    assert atmosphere_path.is_file()
    assert contrast_path.is_file()


def test_exoplanet_discovery_plots_highlight_selected_systems(tmp_path):
    planets = [
        {"pl_name": "HD 108147 b", "hostname": "HD 108147", "discoverymethod": "Radial Velocity",
         "disc_year": "2002", "pl_orbper": "10.9", "pl_rade": "10.9"},
        {"pl_name": "WASP-39 b", "hostname": "WASP-39", "discoverymethod": "Transit",
         "disc_year": "2011", "pl_orbper": "4.05", "pl_rade": "14.3"},
        {"pl_name": "TRAPPIST-1 b", "hostname": "TRAPPIST-1", "discoverymethod": "Transit",
         "disc_year": "2016", "pl_orbper": "1.51", "pl_rade": "1.12"},
        {"pl_name": "unmeasured planet", "hostname": "Unknown", "discoverymethod": "Transit",
         "disc_year": "", "pl_orbper": "", "pl_rade": ""},
    ]
    targets = {
        "HD 108147": "TOI-1233 / HD 108147",
        "WASP-39": "WASP-39",
        "TRAPPIST-1": "TRAPPIST-1",
    }

    timeline_path = plot_exoplanet_discovery_timeline(
        planets, targets, tmp_path / "discovery_timeline.png"
    )
    period_radius_path = plot_exoplanet_period_radius(
        planets, targets, tmp_path / "period_radius.png"
    )

    assert timeline_path.is_file() and timeline_path.stat().st_size > 1000
    assert period_radius_path.is_file() and period_radius_path.stat().st_size > 1000


def test_kepler_third_law_round_trip():
    period = 10.0  # [day]
    mass = 1.0  # [M_sun]
    semi_major_axis = retr_smaxkepl(period, mass)
    recovered_period = retr_perikepl(semi_major_axis, mass)
    assert recovered_period == pytest.approx(period, rel=1e-10)


def test_fold_tser_wraps_and_sorts_phase():
    time = np.array([0.0, 1.5, 3.0, 4.5, 6.0])  # [day]
    flux = np.array([1.0, 0.9, 1.0, 0.9, 1.0])
    data = np.column_stack((time, flux))
    folded = fold_tser(data, epoc=0.0, peri=3.0, boolsort=True)

    assert folded.shape == (5, 2)
    assert np.all(np.diff(folded[:, 0]) >= 0.0)
    assert np.all(folded[:, 0] >= 0.0)
    assert np.all(folded[:, 0] < 1.0)

    folded_centered = fold_tser(data, epoc=0.0, peri=3.0, phascntr=0.0, boolsort=True)
    assert np.all(folded_centered[:, 0] >= -0.5)
    assert np.all(folded_centered[:, 0] < 0.5)


def test_retr_lcurmodl_flarsing_produces_fred_profile():
    time = np.linspace(-1.0, 3.0, 401)  # [day]
    flare = retr_lcurmodl_flarsing(time, timeflar=0.0, amplflar=0.05, scalrise=0.02, scalfall=0.1)

    assert flare.shape == time.shape
    peak_index = np.argmax(flare)
    assert time[peak_index] == pytest.approx(0.0, abs=0.01)
    assert flare[peak_index] == pytest.approx(0.05, rel=1e-3)
    assert flare[time < -0.2].max() < 0.001
    assert flare[time > 2.0].max() < 0.001


def test_planetary_metric_and_orbital_helpers():
    # escape velocity
    assert retr_vesc(1.0, 1.0) == pytest.approx(11.2)
    assert retr_vesc(np.array([1.0, 4.0]), np.array([1.0, 1.0])) == pytest.approx(np.array([11.2, 22.4]))

    # radius ratio to semi-major axis
    assert retr_rs2a(0.105, 0.05) == pytest.approx(0.10)

    # scaled semi-major axis from duration
    assert retr_rsmacomp(10.0, 2.4, 0.0) == pytest.approx(np.sin(np.pi / 100.0))

    # transit duration
    assert retr_duratrantotl(10.0, 0.1, 0.0) == pytest.approx(24.0 * 10.0 / np.pi * np.arcsin(0.1))
    assert retr_duratrantotl(10.0, 0.1, 0.2) == 0.0

    # transmission spectroscopy metric
    tsm = retr_tsmm(2.0, 500.0, 5.0, 1.0, 10.0)
    assert tsm == pytest.approx(1.26 * 8.0 * 500.0 / 5.0 * 0.01)

    # atmospheric scale height
    scale = retr_scalheig(160.0, 1.0, 1.0)
    assert scale == pytest.approx(27.0 / 71398.0)

    # Roche limit and Hill sphere
    assert retr_radiroch(1.0, 1.0, 1.0) == pytest.approx(2.0 ** (1.0 / 3.0))
    assert retr_radihill(1.0, 3e-6, 1.0) == pytest.approx(0.01)

    # surface gravity
    assert retr_logg(1.0, 1.0) == pytest.approx(4.437)

    # impact parameter
    assert retr_imfa(0.1, 0.2, 0.0, 0.0) == pytest.approx(0.5)

    # beaming and ellipsoidal variations
    assert retr_deptbeam(1.0, 1.0, 0.001) > 0.0
    assert retr_deptelli(1.0, 1.4, 1.0, 0.001) > 0.0
    assert retr_alphelli(0.6, 0.4) > 0.0

    # day/night temperatures and emission spectroscopy metric
    t_day, t_night = retr_tmptplandayynigh(1000.0, 0.1)
    assert t_day > t_night
    assert retr_esmm(1000.0, 5778.0, 1.0, 1.0, 10.0) > 0.0

    # semi-amplitude and full orbit radial velocity
    k_semi = keplerian_semi_amplitude(5.0, 1.0, 1e-3, 90.0, 0.0)
    assert k_semi > 0.0
    assert retr_rvelsema(5.0, 1.0, 1e-3, 90.0, 0.0) == pytest.approx(k_semi)

    time_pts = np.array([102.5, 103.75, 105.0])
    rv_curve = keplerian_orbit_radial_velocity(time_pts, 102.5, 5.0, 1e-3, 1.0, 90.0, 0.0, 90.0)
    assert rv_curve[0] == pytest.approx(0.0, abs=1e-9)
    assert retr_rvel(time_pts, 102.5, 5.0, 1e-3, 1.0, 90.0, 0.0, 90.0) == pytest.approx(rv_curve)

    # mass to radius empirical relation
    radii = retr_radifrommass(np.array([1.0, 10.0, 300.0, 50000.0]))
    assert radii[0] == pytest.approx(1.0)
    assert radii[1] > radii[0]
    assert np.all(radii > 0.0)

    # albedo
    albedo = retr_albg(1e-4, 1.0, 0.05)
    assert albedo > 0.0

    # resonance and companion names/colors
    labels = retr_liststrgcomp(4)
    assert list(labels) == ["b", "c", "d", "e"]
    colors = retr_listcolrcomp(3)
    assert len(colors) == 3

    periods = np.array([[10.0, 20.01]])
    intgreso, rati = retr_reso(periods)
    assert intgreso[0, 1, 0] == 2
    assert intgreso[0, 1, 1] == 1

    # limb-darkened surface brightness laws
    assert stellar_limb_darkened_brightness(1.0, (0.4, 0.25), law="quad") == pytest.approx(1.0)
    assert retr_brgtlmdk(0.5, (0.4, 0.25), law="quad") > 0.0
    assert retr_brgtlmdk(0.5, (0.6,), law="linr") == pytest.approx(0.7)
    assert retr_brgtlmdk(0.5, (), law="none") == 1.0

    # delta-magnitude to relative flux
    flux, flux_err = relative_flux_from_delta_magnitude(2.5, uncertainty=0.1)
    assert flux == pytest.approx(0.1)
    assert flux_err > 0.0
    assert retr_rflxfromdmag(0.0) == pytest.approx(1.0)

    # solar system planets
    from tdpy.exoplanet import solar_system_planets, retr_dictsols, keplerian_orbital_speed, retr_vorb

    sols = solar_system_planets()
    assert len(sols["name"]) == 8
    assert sols["name"][2] == "Earth"
    assert sols["mass"][2] == pytest.approx(1.0)
    assert sols["radi"][2] == pytest.approx(1.0)
    assert retr_dictsols()["name"][0] == "Mercury"

    # keplerian orbital speed: ~29.78 km/s at 1 AU around 1 M_sun
    v_earth = keplerian_orbital_speed(1.0, 1.0)
    assert v_earth == pytest.approx(29.7847, rel=1e-3)
    assert retr_vorb(1.0, 1.0) == pytest.approx(v_earth)

    # dilution factor
    from tdpy.exoplanet import companion_flux_dilution, retr_dilu, retr_rflxmodlrise

    dilu = companion_flux_dilution(5778.0, 3500.0)
    assert 0.0 < dilu < 1.0
    assert retr_dilu(5778.0, 3500.0, "tess") == pytest.approx(dilu)

    # quadratic rise profile
    time_rise = np.array([-0.1, 0.0, 0.2])
    rflx, dflx = retr_rflxmodlrise(time_rise, 0.0, 0.1, 0.05)
    assert rflx[0] == 1.0
    assert dflx[0] == 0.0
    assert rflx[1] == 1.0
    assert rflx[2] == pytest.approx(1.0 + 0.1 * 0.2 + 0.05 * 0.04)

    # transit epochs and time indices
    from tdpy.exoplanet import transit_epochs, transit_times, transit_time_indices, retr_indxtran, retr_listepoctran, retr_indxtimetran

    time_series = np.linspace(0.0, 10.0, 1001)  # 0 to 10 days
    epochs = transit_epochs(time_series, epoch_days=1.0, period_days=3.0, duration_hours=2.4)
    assert list(epochs) == [0, 1, 2]
    assert list(retr_indxtran(time_series, 1.0, 3.0, 2.4)) == [0, 1, 2]

    times = transit_times(time_series, epoch_days=1.0, period_days=3.0, duration_hours=2.4)
    np.testing.assert_allclose(times, [1.0, 4.0, 7.0])
    np.testing.assert_allclose(retr_listepoctran(time_series, 1.0, 3.0, 2.4), [1.0, 4.0, 7.0])

    indices = transit_time_indices(time_series, epoch_days=1.0, period_days=3.0, duration_hours=2.4)
    assert indices.size > 0
    # Center of transit at t=1.0 day should be in transit indices
    center_idx = np.argmin(np.abs(time_series - 1.0))
    assert center_idx in indices

    out_indices = transit_time_indices(time_series, epoch_days=1.0, period_days=3.0, duration_hours=2.4, out_of_transit=True)
    assert center_idx not in out_indices
    assert len(indices) + len(out_indices) == len(time_series)
