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
```

A legacy setup-based install still works for compatibility, but the modern editable install is preferred for reproducible development.

## Quick example

The central synthetic example exercises the catalog-plotting path used by multiple time-domain workflows:

```python
from tdpy.examples.synthetic_catalog_demo import run_demo

path = run_demo(output_dir='.')
print(path)
```

This produces a small on-disk figure showing the input mock catalog and the diagnostic catalog overlay. It is intentionally toy data with clearly stated generative assumptions, so it demonstrates the plotting logic without pretending to represent real scientific evidence.

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

## Testing status

The package has light-weight regression tests covering:

- normalized environment-backed path handling;
- catalog-plot annotation behavior;
- the synthetic example workflow.

## Development status

This repository is maintained as a shared infrastructure layer rather than a project-specific analysis package. Its scientific value is in being reusable, transparent, and stable enough for other astrophysics repositories to build on.


