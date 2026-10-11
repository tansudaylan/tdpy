import matplotlib.colors
import numpy as np

import tdpy


def test_make_cmap_preserves_endpoints():
    """Custom colormaps interpolate between the requested endpoint colors."""

    cmap = tdpy.make_cmap([(1., 0., 0.), (0., 0., 1.)])

    np.testing.assert_allclose(cmap(0.)[:3], (1., 0., 0.))
    np.testing.assert_allclose(cmap(1.)[:3], (0., 0., 1.))


def test_make_cmapdivg_is_white_centered():
    """Diverging colormaps preserve named endpoints and a white midpoint."""

    cmap = tdpy.make_cmapdivg('Red', 'Orange')

    np.testing.assert_allclose(cmap(0.)[:3], matplotlib.colors.to_rgb('Red'))
    np.testing.assert_allclose(cmap(0.5)[:3], (1., 1., 1.), atol=0.01)
    np.testing.assert_allclose(cmap(1.)[:3], matplotlib.colors.to_rgb('Orange'))


def test_washu_palette_is_exported():
    assert tdpy.WASHU_RED == "#A51417"
    assert tdpy.WASHU_GREEN == "#007360"
    assert tdpy.WASHU_GRAY == "#6C7373"
    assert tdpy.WASHU_COLORS == {
        "red": tdpy.WASHU_RED,
        "green": tdpy.WASHU_GREEN,
        "gray": tdpy.WASHU_GRAY,
    }
