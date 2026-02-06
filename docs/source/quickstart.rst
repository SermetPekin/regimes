Quick Start Guide
=================

This guide will help you get started with regimes.

Installation
------------

Install regimes using pip:

.. code-block:: bash

   pip install regimes

Testing for Structural Breaks
-----------------------------

The primary use case is testing for structural breaks in time series data
using the Bai-Perron test:

.. code-block:: python

   import numpy as np
   import regimes as rg

   # Create sample data with a break
   np.random.seed(42)
   y = np.concatenate([
       np.random.randn(100),      # regime 1
       np.random.randn(100) + 2,  # regime 2
   ])

   # Run Bai-Perron test
   test = rg.BaiPerronTest(y)
   results = test.fit(max_breaks=5)

   # View results
   print(results.summary())

The results include:

- **Sup-F statistics**: Test m breaks vs 0 breaks
- **UDmax**: Test for presence of any breaks
- **BIC/LWZ**: Information criteria for model selection
- **Break locations**: Optimal break points for each number of breaks

Regression with HAC Standard Errors
-----------------------------------

For regression models with autocorrelated errors:

.. code-block:: python

   import numpy as np
   import regimes as rg

   # Generate data
   n = 200
   X = np.column_stack([np.ones(n), np.random.randn(n)])
   y = X @ [1, 2] + np.random.randn(n)

   # Fit OLS with Newey-West standard errors
   model = rg.OLS(y, X)
   results = model.fit(cov_type="HAC")

   print(results.summary())

AR Models with Breaks
---------------------

For autoregressive models with known structural breaks:

.. code-block:: python

   import numpy as np
   import regimes as rg

   # Simulate AR(1) process
   n = 200
   y = np.zeros(n)
   for t in range(1, n):
       y[t] = 0.7 * y[t-1] + np.random.randn()

   # Fit AR(1) with break at t=100
   model = rg.AR(y, lags=1, breaks=[100])
   results = model.fit(cov_type="HAC")

   print(results.summary())

Visualization
-------------

Plot your results:

.. code-block:: python

   import regimes as rg

   # Basic break plot
   fig, ax = rg.plot_breaks(y, breaks=[100])

   # With regime shading
   fig, ax = rg.plot_breaks(y, breaks=[100], shade_regimes=True)

   # Regime means
   fig, ax = rg.plot_regime_means(y, breaks=[100])
