"""Public API for regimes package.

This module provides a clean namespace for the most commonly used
classes and functions in the regimes package.
"""

# Models
from regimes.models import (
    ADL,
    ADLResults,
    AR,
    ARResults,
    OLS,
    OLSResults,
    adl_summary_by_regime,
    ar_summary_by_regime,
    summary_by_regime,
)

# Diagnostics
from regimes.diagnostics import DiagnosticTestResult, DiagnosticsResults

# Tests
from regimes.tests import BaiPerronResults, BaiPerronTest

# Visualization
from regimes.visualization import (
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
)

# Rolling/Recursive estimation
from regimes.rolling import (
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
)

# Results base classes (for type checking)
from regimes.results import (
    BreakResultsBase,
    RegressionResultsBase,
    RegimesResultsBase,
)

# Model base classes (for extension)
from regimes.models import (
    CovType,
    RegimesModelBase,
    TimeSeriesModelBase,
)

__all__ = [
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
