import os

import matplotlib
matplotlib.use('Agg')


def test_synthetic_catalog_demo_runs():
    from tdpy.examples.synthetic_catalog_demo import run_demo

    path = run_demo(output_dir='.')
    assert os.path.exists(path)
