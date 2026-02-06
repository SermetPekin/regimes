regimes documentation
=========================

**regimes** is a Python package for structural break detection and estimation
in time-series econometrics, extending statsmodels with robust methods for
analyzing regime changes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api

Features
--------

- **Structural Break Tests**: Bai-Perron test for multiple structural breaks
- **Time-Series Models**: AR, OLS with HAC standard errors and known break support
- **Model Selection**: BIC, LWZ criteria for selecting the number of breaks
- **Visualization**: Plot time series with break lines and confidence intervals

Installation
------------

.. code-block:: bash

   pip install regimes

Quick Example
-------------

.. code-block:: python

   import numpy as np
   import regimes as rg

   # Simulate data with a mean shift
   y = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])

   # Test for breaks
   test = rg.BaiPerronTest(y)
   results = test.fit(max_breaks=3)
   print(f"Detected {results.n_breaks} break(s) at: {results.break_indices}")


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
