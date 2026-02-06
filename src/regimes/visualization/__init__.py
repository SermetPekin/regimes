"""Visualization utilities for structural break analysis."""

from regimes.visualization.breaks import (
    plot_break_confidence,
    plot_breaks,
    plot_regime_means,
)
from regimes.visualization.diagnostics import (
    plot_actual_fitted,
    plot_diagnostics,
    plot_residual_acf,
    plot_residual_distribution,
    plot_scaled_residuals,
)
from regimes.visualization.params import plot_params_over_time
from regimes.visualization.rolling import plot_rolling_coefficients
from regimes.visualization.style import (
    REGIMES_COLOR_CYCLE,
    REGIMES_COLORS,
    add_break_dates,
    add_confidence_band,
    add_source,
    get_style,
    label_line_end,
    set_style,
    shade_regimes,
    use_style,
)

__all__ = [
    # Plotting functions
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
    # Style utilities
    "REGIMES_COLORS",
    "REGIMES_COLOR_CYCLE",
    "get_style",
    "set_style",
    "use_style",
    "label_line_end",
    "add_break_dates",
    "add_confidence_band",
    "shade_regimes",
    "add_source",
]
