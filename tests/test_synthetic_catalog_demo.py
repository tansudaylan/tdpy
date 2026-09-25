import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
from pathlib import Path

from tdpy.catalog import plot_synthetic_catalog_diagnostic


def test_synthetic_catalog_demo_runs(tmp_path, capsys):
    output_path = tmp_path / "catalog_overlay_diagnostic.png"

    path = plot_synthetic_catalog_diagnostic(output_path)

    image = mpimg.imread(path)
    assert Path(path) == output_path
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert f"Writing to {output_path}..." in capsys.readouterr().out
