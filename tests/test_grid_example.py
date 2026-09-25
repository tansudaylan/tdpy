import runpy
from pathlib import Path

import tdpy


def test_grid_example_import_does_not_plot(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Importing the grid example must not create plots.")

    monkeypatch.setattr(tdpy, "plot_grid", fail_if_called)
    example_path = Path(__file__).parents[1] / "examples" / "test_grid.py"
    runpy.run_path(str(example_path), run_name="tdpy_grid_example")