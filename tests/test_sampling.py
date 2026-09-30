import pickle
import inspect

import numpy as np

from tdpy import mcmc
import tdpy.mcmc_depr as deprecated_mcmc
import tdpy.util as util
from tdpy.mcmc_depr import samp as deprecated_samp
from tdpy.util import retr_lpos, samp


def test_periodic_score_sensitivity_is_invertible():
    """Known-phase event requirements invert the semiamplitude sensitivity."""
    amplitude = 0.5
    event_count = util.periodic_score_event_count(amplitude)

    assert 129 < event_count < 131
    np.testing.assert_allclose(util.periodic_score_amplitude(event_count), amplitude)


def test_mcmc_gmrb_uses_shared_implementation():
    """The compatibility MCMC module exposes the canonical convergence helper."""

    griddata = np.array([[0., 1., 2.], [1., 2., 4.], [2., 4., 5.]])

    assert mcmc.gmrb_test is util.gmrb_test
    assert mcmc.plot_gmrb is util.plot_gmrb
    np.testing.assert_allclose(mcmc.gmrb_test(griddata), util.gmrb_test(griddata))


def test_mcmc_autocorrelation_uses_shared_implementation():
    """Autocorrelation compatibility paths preserve both verbosity keywords."""

    samples = np.arange(24., dtype=float).reshape(8, 3)

    assert mcmc.retr_atcr_neww is util.retr_atcr_neww
    assert mcmc.retr_timeatcr is util.retr_timeatcr
    assert mcmc.plot_atcr is util.plot_atcr
    result_new = mcmc.retr_timeatcr(samples, typeverb=0)
    result_legacy = mcmc.retr_timeatcr(samples, verbtype=0)
    np.testing.assert_allclose(result_new[0], result_legacy[0])
    assert result_new[1] == result_legacy[1]


def test_deprecated_transforms_use_shared_implementations():
    """The deprecated MCMC module reuses canonical transform helpers directly."""

    names = (
        "icdf_self",
        "icdf_logt",
        "icdf_atan",
        "icdf_gaus",
        "cdfn_self",
        "cdfn_logt",
        "cdfn_atan",
        "cdfn_samp",
        "cdfn_samp_sing",
        "icdf_samp",
        "icdf_samp_sing",
        "retr_numbsamp",
        "retr_icdfunif",
        "retr_icdf",
    )

    for name in names:
        assert getattr(deprecated_mcmc, name) is getattr(util, name)


def test_retr_lpos_penalizes_both_gaussian_tails():
    def retr_llik(para, gdat):
        return 0.

    args = [None, np.arange(1), ['gaus'], np.array([-10.]), np.array([10.]),
            np.array([0.]), np.array([1.]), retr_llik, None]
    values = [retr_lpos(np.array([value]), *args) for value in (-1., 0., 1.)]

    assert np.allclose(values, [-0.5, 0., -0.5])


def test_pcat_evidence_preserves_constant_likelihood(tmp_path):
    def retr_llik(para, gdat):
        return np.log(3.)

    chain, logprob, evidence = util._pcat_legacy_chains(
        None, retr_llik, None, ['x'], ['self'], np.array([0.]), np.array([1.]),
        None, None, np.array([[0.5]]), 1, 8, 2, str(tmp_path), -1,
        estimate_log_evidence=True, evidence_samples=300, seed=7,
    )
    assert chain.shape == (1, 8, 1)
    assert logprob.shape == (1, 8)
    assert abs(evidence['log_evidence'] - np.log(3.)) < 4 * evidence['relative_error']
    assert evidence['relative_error'] < 0.1


def test_allesfitter_adapter_writes_pcat_chain_for_existing_reader(tmp_path, monkeypatch):
    import sys
    import types
    import h5py

    config = types.ModuleType('allesfitter.config')
    config.init = lambda path: setattr(config, 'BASEMENT', types.SimpleNamespace(
        datadir=path, bounds=[('uniform', 0., 1.), ('normal', 0., 1.)],
        theta_0=np.array([0.5, 0.]), outdir=str(tmp_path / 'results'),
        settings={'mcmc_nwalkers': 2, 'mcmc_total_steps': 12, 'mcmc_thin_by': 2},
    ))
    allesfitter = types.ModuleType('allesfitter')
    allesfitter.config = config
    monkeypatch.setitem(sys.modules, 'allesfitter', allesfitter)
    monkeypatch.setitem(sys.modules, 'allesfitter.config', config)

    def fake_pcat_chains(*args):
        assert args[4] == ['self', 'gaus']
        return np.ones((2, 6, 2)), np.zeros((2, 6))

    monkeypatch.setattr(util, '_pcat_legacy_chains', fake_pcat_chains)
    path = util.sample_allesfitter_pcat(str(tmp_path))
    print('Reading from %s...' % path)
    with h5py.File(path, 'r') as saved:
        assert saved['mcmc/chain'].shape == (6, 2, 2)
        assert saved['mcmc/log_prob'].shape == (6, 2)
        assert saved['mcmc'].attrs['iteration'] == 6


def test_deprecated_samp_allows_valid_burn_in():

    def retr_llik(para, gdat):
        return 0.

    result, _ = deprecated_samp(
        None, None, 10, retr_llik, ['x'], [['X', '']], ['self'],
        np.array([0.]), np.array([10.]), numbsampburnwalk=3, boolmult=False,
        diagmode=False, verbtype=0,
    )

    assert np.asarray(result['x']).size == 20 * 7
    assert np.all(np.isfinite(result['x']))


def test_deprecated_nested_mode_uses_pcat_evidence(tmp_path):
    def retr_llik(para, gdat):
        return np.log(3.)

    parameters, derived = deprecated_samp(
        None, None, 8, retr_llik, ['x'], [['X', '']], ['self'],
        np.array([0.]), np.array([1.]), typesamp='nest',
        numbsampburnwalk=2, verbtype=-1,
    )
    assert len(parameters['x']) == 20 * 6
    assert abs(derived['log_evidence'] - np.log(3.)) < 0.1
    assert derived['log_evidence_relative_error'] < 0.1


def test_samp_clips_retained_post_burn_in_samples_to_available_data(monkeypatch):
    def fake_pcat_chains(*args):
        initial, numbwalk, numbsampwalk = args[9:12]
        assert np.all(np.isfinite(initial))
        chain = np.arange(numbwalk)[:, None, None] + np.arange(numbsampwalk)[None, :, None]
        return chain.astype(float), np.zeros((numbwalk, numbsampwalk))

    monkeypatch.setattr(util, '_pcat_legacy_chains', fake_pcat_chains)

    def retr_llik(para, gdat):
        return 0.

    result = samp(
        None, 5, retr_llik, ['x'], [['X', '']], ['self'], np.array([0.]), np.array([1.]),
        numbsampburnwalk=3, numbsamppostwalk=20, booltqdm=False, typeverb=0,
    )

    assert result['x'].size == 20 * (5 - 3)
    assert result['x'][0] == 3.
    assert result['x'][-1] == 23.


def test_samp_aligns_derived_results_and_summarizes_each_parameter(monkeypatch, tmp_path):
    def fake_pcat_chains(*args):
        initial, numbwalk, numbsampwalk = args[9:12]
        initial = np.asarray(initial)
        assert np.all(np.isfinite(initial))
        numbpara = initial.shape[1]
        if numbpara == 2:
            assert np.any(initial[:, 0] < 0.)
        chain = np.empty((numbwalk, numbsampwalk, numbpara))
        chain[:, :, 0] = np.arange(numbwalk)[:, None]
        if numbpara == 2:
            chain[:, :, 1] = 100. + np.arange(numbsampwalk)[None, :]
        return chain, np.zeros((numbwalk, numbsampwalk))

    monkeypatch.setattr(util, '_pcat_legacy_chains', fake_pcat_chains)
    plot_calls = []
    plot_signature = inspect.signature(util.plot_grid)

    def check_plot_call(*args, **kwargs):
        plot_signature.bind(*args, **kwargs)
        plot_calls.append(kwargs)

    monkeypatch.setattr(util, 'plot_grid', check_plot_call)
    monkeypatch.setattr(util.plt, 'savefig', lambda path: None)

    def retr_llik(para, gdat):
        return 0.

    def retr_dictderi(para, gdat):
        return {'sum': para.sum(), 'pair': para.copy()}

    output_path = tmp_path / 'output with spaces'
    pathbase = str(output_path) + '/'
    result = samp(
        None, 3, retr_llik, ['gaussian', 'uniform'], [['Gaussian', ''], ['Uniform', '']],
        ['gaus', 'self'], np.array([-10., 0.]), np.array([10., 200.]), pathbase=pathbase,
        numbsamppostwalk=3, meangauspara=np.array([0., 0.]), stdvgauspara=np.array([1., 1.]),
        retr_dictderi=retr_dictderi, dictlablscalparaderi={'sum': ['Sum', '']}, boolplot=True,
        booltqdm=False, typeverb=0,
    )

    assert result['sum'].shape == (60,)
    assert result['pair'].shape == (60, 2)
    assert np.allclose(result['sum'], result['pair'].sum(axis=1))

    assert any(call.get('listnamepara') == ['sum'] for call in plot_calls)

    pathdata = output_path / 'mcmc' / 'data'
    summary = np.loadtxt(pathdata / 'postpara.csv', delimiter=',')
    samples = np.column_stack((result['gaussian'], result['uniform'], result['sum']))
    assert np.allclose(summary[:, 0], np.median(samples, axis=0))
    assert np.allclose(summary[:, 1], np.percentile(samples, 84., axis=0) - summary[:, 0])
    assert np.allclose(summary[:, 2], summary[:, 0] - np.percentile(samples, 16., axis=0))

    with (pathdata / 'postderi.pickle').open('rb') as fileobj:
        saved = pickle.load(fileobj)
    assert saved['sum'].shape == (60,)
    assert saved['pair'].shape == (60, 2)

    result_memory = samp(
        None, 3, retr_llik, ['uniform'], [['Uniform', '']], ['self'], np.array([0.]),
        np.array([1.]), numbsamppostwalk=3, booltqdm=False, typeverb=0,
    )
    assert result_memory['uniform'].shape == (60,)