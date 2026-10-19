Getting started
===============

Installation
------------

Install TDpy from its repository root:

.. code-block:: console

   python -m pip install -e .

Set ``TDPY_PATH`` to the repository root when a helper reads local data or
writes products there. Most numerical routines accept arrays and require no
repository-local files.

Example
-------

The target-visibility workflow uses a named observatory and target:

.. code-block:: console

   python examples/target_visibility/run.py --target TOI-1233 --observatory TUG

It computes one-night altitude curves and annual target visibility from
coordinates and observatory metadata. No observation data are downloaded.

Module layout
-------------

``tdpy.numerics`` contains distribution transforms, array summaries, covariance
helpers, and forecasts. ``tdpy.astro`` contains target geometry and observing
calculations. ``tdpy.tess`` creates TESS footprint and visibility products.
``tdpy.serialization`` handles structured workflow output, and
``tdpy.workflow`` runs labeled fail-fast command sequences.