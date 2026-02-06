"""regimes: Structural break detection and estimation for time-series econometrics.

A Python package extending statsmodels with structural break detection
and estimation capabilities for econometric time-series analysis.

Example
-------
>>> import regimes as rg
>>> import numpy as np
>>>
>>> # Simulate data with one break
>>> np.random.seed(42)
>>> y = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])
>>>
>>> # Test for breaks using Bai-Perron
>>> test = rg.BaiPerronTest(y)
>>> results = test.fit(max_breaks=3)
>>> print(f"Detected {results.n_breaks} break(s) at: {results.break_indices}")
>>>
>>> # Fit AR model with known break
>>> ar_model = rg.AR(y, lags=1, breaks=[100])
>>> ar_results = ar_model.fit(cov_type="HAC")
>>> print(ar_results.summary())
>>>
>>> # Fit ADL model
>>> x = np.random.randn(len(y))
>>> adl_model = rg.ADL(y, x, lags=1, exog_lags=1)
>>> adl_results = adl_model.fit()
>>> print(adl_results.summary())
"""

from regimes._version import __version__
from regimes.api import (
    # Models
    ADL,
    ADLResults,
    AR,
    ARResults,
    OLS,
    OLSResults,
    adl_summary_by_regime,
    ar_summary_by_regime,
    summary_by_regime,
    # Diagnostics
    DiagnosticTestResult,
    DiagnosticsResults,
    # Tests
    BaiPerronResults,
    BaiPerronTest,
    # Rolling/Recursive estimation
    RecursiveADL,
    RecursiveADLResults,
    RecursiveAR,
    RecursiveARResults,
    RecursiveOLS,
    RecursiveOLSResults,
    RollingADL,
    RollingADLResults,
    RollingAR,
    RollingARResults,
    RollingCovType,
    RollingEstimatorBase,
    RollingOLS,
    RollingOLSResults,
    RollingResultsBase,
    # Visualization
    plot_actual_fitted,
    plot_break_confidence,
    plot_breaks,
    plot_diagnostics,
    plot_params_over_time,
    plot_regime_means,
    plot_residual_acf,
    plot_residual_distribution,
    plot_rolling_coefficients,
    plot_scaled_residuals,
    # Base classes
    BreakResultsBase,
    CovType,
    RegressionResultsBase,
    RegimesModelBase,
    RegimesResultsBase,
    TimeSeriesModelBase,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "OLS",
    "OLSResults",
    "summary_by_regime",
    "AR",
    "ARResults",
    "ar_summary_by_regime",
    "ADL",
    "ADLResults",
    "adl_summary_by_regime",
    # Diagnostics
    "DiagnosticTestResult",
    "DiagnosticsResults",
    # Tests
    "BaiPerronTest",
    "BaiPerronResults",
    # Rolling/Recursive estimation
    "RollingOLS",
    "RollingOLSResults",
    "RecursiveOLS",
    "RecursiveOLSResults",
    "RollingAR",
    "RollingARResults",
    "RecursiveAR",
    "RecursiveARResults",
    "RollingADL",
    "RollingADLResults",
    "RecursiveADL",
    "RecursiveADLResults",
    "RollingEstimatorBase",
    "RollingResultsBase",
    "RollingCovType",
    # Visualization
    "plot_breaks",
    "plot_regime_means",
    "plot_break_confidence",
    "plot_params_over_time",
    "plot_rolling_coefficients",
    "plot_diagnostics",
    "plot_actual_fitted",
    "plot_scaled_residuals",
    "plot_residual_distribution",
    "plot_residual_acf",
    # Base classes
    "RegimesModelBase",
    "TimeSeriesModelBase",
    "RegimesResultsBase",
    "RegressionResultsBase",
    "BreakResultsBase",
    "CovType",
]
