import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import pytest


EXAMPLE_PATH = (
    Path(__file__).parents[1] / "examples" / "psf_profiles" / "double_king_profile.py"
)
SPEC = importlib.util.spec_from_file_location(
    "double_king_compatibility", EXAMPLE_PATH
)
double_king_compatibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(double_king_compatibility)


def test_double_king_compatibility_example_writes_exact_comparison(tmp_path):
    output_path = tmp_path / "double_king_compatibility.png"

    maximum_difference = double_king_compatibility.run_example(output_path)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert maximum_difference == 0.0
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8


def test_double_king_compatibility_example_writes_bokeh_html(tmp_path):
    pytest.importorskip("bokeh")
    output_path = double_king_compatibility.run_bokeh_example(
        tmp_path / "double_king.html"
    )

    assert output_path.is_file()
    assert "TDpy double-King profile" in output_path.read_text(encoding="utf-8")