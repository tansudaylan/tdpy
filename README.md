# tdpy — shared numerical and visualization utilities

## Scientific purpose

`tdpy` is the shared numerical and plotting foundation for the active astrophysics ecosystem. It provides the reusable routines that keep the scientific workflows inspectable and portable: environment-aware path handling, synthetic and real-data diagnostics, plotting conventions, and foundational numerical helpers used across multiple repositories.

This library is intended to hold functionality that is intentionally generic enough to be reused by many astrophysical workflows without becoming a monolithic project-specific analysis script.

## What it provides

The library includes utilities for:

- environment-based filesystem and data-path normalization;
- reusable plotting helpers for population and catalog diagnostics;
- numerical support for array handling and scientific transforms;
- light-weight visualization utilities used across time-domain and catalog workflows;
- reproducible output layout for figures and intermediate diagnostic products.

## Current ecosystem role

Within the broader stack, `tdpy` is the canonical shared layer for:

- shared plotting conventions;
- generic astronomy utilities;
- normalized data-directory setup;
- diagnostics that make intermediate workflow states visible;
- low-level numerical support that downstream repositories can reuse instead of reimplementing.

## Installation

The repository supports a standard editable install:

```bash
cd tdpy
python -m pip install -e .
export TDPY_PATH=/path/to/tdpy
```

`TDPY_PATH` identifies the repository root. Runtime inputs belong under `data/` and generated pipeline outputs belong under `visuals/`. Both directories are ignored by Git. Dataset-specific helpers continue to use their established `*_DATA_PATH` variables.

A legacy setup-based install still works for compatibility, but the modern editable install is preferred for reproducible development.

## Quick example

The central synthetic example exercises the catalog-plotting path used by multiple time-domain workflows:

```bash
python examples/catalog_overlay_diagnostic.py --typefileplot png
```

![Synthetic catalog overlay diagnostic](examples/catalog_overlay_diagnostic.png)

This produces a two-panel figure showing a deterministic Gaussian source field before and after catalog annotation. The three sources have mean magnitudes of 11, 12, and 13 mag. Two are isolated and one is labeled as blended. These are clearly labeled simulated inputs rather than observational evidence.

## Example workflow

The demonstration follows the intended project pattern:

- input: synthetic catalog positions and magnitudes;
- transformation: catalog overlay plotting and annotation placement;
- output: a saved figure showing the diagnostic field view.

This keeps the scientific reasoning visible without burying calculations inside a monolithic plotting script.

## Core plotting utilities

`tdpy.plot_grid()` is the larger figure-generation entry point for parameter-grid and population diagnostics, while `tdpy.plot_catl()` is the compact catalog-plotting diagnostics helper used for structured field overlays and source annotations.

These routines are designed to be reused by scientific workflows rather than copied into individual repositories.

## Path conventions

The package includes helper functions for normalized data paths:

- `retr_pathbase()`
- `retr_pathenv()`
- `ensr_path()`
- `retr_path()`

These helpers provide a consistent pattern for environment-backed data directories while avoiding hard-coded personal filesystem paths.


