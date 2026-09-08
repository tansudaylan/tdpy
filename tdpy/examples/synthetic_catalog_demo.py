"""Minimal synthetic demonstration of the catalog plotting workflow.

This intentionally uses a toy catalog with clearly stated generative assumptions:
- a small grid of mock sources with known positions and magnitudes;
- catalog labels are overplotted on a synthetic field;
- the code exits with a save path so the workflow can be checked in tests.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tdpy


class DummyCatalog:
    """Small synthetic object that exercises tdpy.plot_catl()."""

    def __init__(self):
        self.numbpositext = 3
        self.indxsideyposdataflat = np.array([10.0, 20.0, 30.0])
        self.indxsidexposdataflat = np.array([12.0, 24.0, 36.0])
        self.indxdatascorsort = np.array([0, 1, 2])
        self.numbsideedge = 3
        self.datatype = 'mock'
        self.indxsour = np.array([0, 1, 2])
        self.indxsoursupn = np.array([0, 1, 2])
        self.trueypos = np.array([[5.0, 7.0, 9.0], [11.0, 13.0, 15.0]])
        self.truexpos = np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])
        self.truemagtmean = np.array([11.0, 12.0, 13.0])
        self.truemagtstdv = np.array([0.1, 0.2, 0.3])


def run_demo(output_dir='.'):
    """Create a small catalog-plot output and return the saved path."""

    catalog = DummyCatalog()
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.set_xlim(0, 40)
    axis.set_ylim(0, 40)
    axis.set_xlabel('x position')
    axis.set_ylabel('y position')
    axis.set_title('Synthetic catalog mock field')

    tdpy.plot_catl(catalog, axis, indxsideyposoffs=1, indxsidexposoffs=2)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'synthetic_catalog_demo.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return path


if __name__ == '__main__':
    print(run_demo())
