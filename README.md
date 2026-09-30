# TDpy numerical and visualization utilities

## Purpose

TDpy provides numerical, plotting, and path-handling utilities for astrophysical analysis. Users can normalize environment-based data paths, transform scientific arrays, annotate source catalogs, and create reproducible diagnostic figures from simulated or observed data.

The functions are designed for direct use in scripts, notebooks, and larger analysis pipelines.

## What it provides

The library includes utilities for:

- environment-based filesystem and data-path normalization;
- reusable plotting helpers for population and catalog diagnostics;
- numerical support for array handling and scientific transforms;
- light-weight visualization utilities used across time-domain and catalog workflows;
- reproducible output layout for figures and intermediate diagnostic products.

## Scientific utilities

- configure portable data and output paths;
- annotate source catalogs and visualize population parameters;
- inspect intermediate analysis states with diagnostic plots;
- apply general numerical transforms used in astronomical analyses.
- build periodic-event timing and overlap reports with `tdpy.astro`;
- calculate interval contrasts and render atmosphere forecast figures with `tdpy.exoplanet`;
- construct normalized NIRSpec calculations with `tdpy.pandeia`;
- run labeled, fail-fast command sequences with `tdpy.workflow`.

## Installation

The repository supports a standard editable install:

```bash
cd tdpy
python -m pip install -e .
export TDPY_PATH=/path/to/tdpy
```

`TDPY_PATH` identifies the repository root. Runtime inputs belong under `data/` and generated pipeline outputs belong under `visuals/`. Both directories are ignored by Git. Dataset-specific helpers continue to use their established `*_DATA_PATH` variables.

## Quick example

The synthetic example annotates a three-source catalog on a Gaussian image:

```bash
python examples/catalog_overlay_diagnostic.py --typefileplot png
```

![Synthetic catalog overlay diagnostic](examples/catalog_overlay_diagnostic.png)

This produces a two-panel figure showing a deterministic Gaussian source field before and after catalog annotation. The three sources have mean magnitudes of 11, 12, and 13 mag. Two are isolated and one is labeled as blended. These are clearly labeled simulated inputs rather than observational evidence.

## Catalog annotation

The calculation contains:

- input: synthetic catalog positions and magnitudes;
- transformation: catalog overlay plotting and annotation placement;
- output: a saved figure showing the diagnostic field view.

The side-by-side field views expose the catalog positions, magnitudes, blend labels, and annotation placement.

## Core plotting utilities

`tdpy.plot_catl()` overlays source positions, magnitudes, and annotations on image fields. Corner plots of parameter samples are drawn by PCAT; pass `pcat.plot_population_grid` as `plot_posterior` to `tdpy.samp()` to plot its joint posteriors.

These routines can be called directly wherever an analysis needs consistent catalog annotations.

## Data paths

TDpy normalizes environment-backed input and output paths with:

- `retr_pathbase()`
- `retr_pathenv()`
- `ensr_path()`
- `retr_path()`

These functions resolve data directories and create requested output directories without workstation-specific absolute paths.


