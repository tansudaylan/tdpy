import pickle
import inspect
import sys
import types

import numpy as np

from tdpy import mcmc
import tdpy.util as util
from tdpy.mcmc_depr import samp as deprecated_samp
from tdpy.util import retr_lpos, samp


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


def test_retr_lpos_penalizes_both_gaussian_tails():
    def retr_llik(para, gdat):
        return 0.

    args = [None, np.arange(1), ['gaus'], np.array([-10.]), np.array([10.]),
            np.array([0.]), np.array([1.]), retr_llik, None]
    values = [retr_lpos(np.array([value]), *args) for value in (-1., 0., 1.)]

    assert np.allclose(values, [-0.5, 0., -0.5])


def test_deprecated_samp_allows_valid_burn_in(monkeypatch):
    class FakeSampler:
        def __init__(self, numbwalk, numbpara, function, args, pool):
            self.numbwalk = numbwalk
            self.numbpara = numbpara

        def run_mcmc(self, parainit, numbsampwalk, progress):
            parainit = np.asarray(parainit)
            assert np.all(np.isfinite(parainit))
            self.chain = np.empty((self.numbwalk, numbsampwalk, self.numbpara))
            self.chain[:, :, 0] = np.arange(self.numbwalk)[:, None]
            self.lnprobability = np.zeros((self.numbwalk, numbsampwalk))

    monkeypatch.setitem(sys.modules, 'emcee', types.SimpleNamespace(EnsembleSampler=FakeSampler))

    def retr_llik(para, gdat):
        return 0.

    result, _ = deprecated_samp(
        None, None, 10, retr_llik, ['x'], [['X', '']], ['self'],
        np.array([0.]), np.array([10.]), numbsampburnwalk=3, boolmult=False,
        diagmode=False, verbtype=0,
    )

    assert np.asarray(result['x']).size == 20 * 7


def test_samp_clips_retained_post_burn_in_samples_to_available_data(monkeypatch):
    class FakeSampler:
        def __init__(self, numbwalk, numbpara, function, args, pool):
            self.numbwalk = numbwalk
            self.numbpara = numbpara

        def run_mcmc(self, parainit, numbsampwalk, progress):
            parainit = np.asarray(parainit)
            assert np.all(np.isfinite(parainit))
            self.chain = np.empty((self.numbwalk, numbsampwalk, self.numbpara))
            self.chain[:, :, 0] = np.arange(self.numbwalk)[:, None] + np.arange(numbsampwalk)[None, :]
            self.lnprobability = np.zeros((self.numbwalk, numbsampwalk))

    monkeypatch.setitem(sys.modules, 'emcee', types.SimpleNamespace(EnsembleSampler=FakeSampler))

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
    class FakeSampler:
        def __init__(self, numbwalk, numbpara, function, args, pool):
            self.numbwalk = numbwalk
            self.numbpara = numbpara

        def run_mcmc(self, parainit, numbsampwalk, progress):
            parainit = np.asarray(parainit)
            assert np.all(np.isfinite(parainit))
            if self.numbpara == 2:
                assert np.any(parainit[:, 0] < 0.)
            self.chain = np.empty((self.numbwalk, numbsampwalk, self.numbpara))
            self.chain[:, :, 0] = np.arange(self.numbwalk)[:, None]
            if self.numbpara == 2:
                self.chain[:, :, 1] = 100. + np.arange(numbsampwalk)[None, :]
            self.lnprobability = np.zeros((self.numbwalk, numbsampwalk))

    monkeypatch.setitem(sys.modules, 'emcee', types.SimpleNamespace(EnsembleSampler=FakeSampler))
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