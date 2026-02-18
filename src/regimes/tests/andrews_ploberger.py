"""Andrews-Ploberger test for a single structural break at unknown date.

This module implements the Andrews (1993) SupF test and the Andrews-Ploberger
(1994) ExpF and AveF tests. All three statistics are computed from the same
sequence of F-statistics evaluated at every candidate break date in a trimmed
range.

- **SupF**: max F(τ) — powerful against a single sharp break
- **ExpF**: ln[(1/T*) Σ exp(½ F(τ))] — optimal Bayesian average power
- **AveF**: (1/T*) Σ F(τ) — simple average, detects diffuse instability

References
----------
Andrews, D. W. K. (1993). Tests for parameter instability and structural
    change with unknown change point. Econometrica, 61(4), 821-856.

Andrews, D. W. K. & Ploberger, W. (1994). Optimal tests when a nuisance
    parameter is present only under the alternative. Econometrica, 62(6),
    1383-1414.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from regimes.tests.base import BreakTestBase, BreakTestResultsBase

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from regimes.models.adl import ADL
    from regimes.models.ar import AR
    from regimes.models.ols import OLS

# ---------------------------------------------------------------------------
# Critical value tables
# ---------------------------------------------------------------------------
# Source: Andrews (1993) Table 1, Andrews & Ploberger (1994) Table 1.
# Keys: (q, pi0) where q = number of breaking parameters, pi0 = trimming.
# Values: {significance_level: critical_value}.

_SUPF_CRITICAL_VALUES: dict[tuple[int, float], dict[float, float]] = {
    # q=1
    (1, 0.05): {0.10: 11.47, 0.05: 13.97, 0.01: 19.30},
    (1, 0.10): {0.10: 9.63, 0.05: 11.70, 0.01: 16.19},
    (1, 0.15): {0.10: 8.68, 0.05: 10.55, 0.01: 14.51},
    (1, 0.20): {0.10: 8.06, 0.05: 9.63, 0.01: 12.82},
    # q=2
    (2, 0.05): {0.10: 13.78, 0.05: 16.16, 0.01: 21.42},
    (2, 0.10): {0.10: 11.80, 0.05: 13.86, 0.01: 18.42},
    (2, 0.15): {0.10: 10.69, 0.05: 12.57, 0.01: 16.56},
    (2, 0.20): {0.10: 10.01, 0.05: 11.68, 0.01: 15.06},
    # q=3
    (3, 0.05): {0.10: 15.85, 0.05: 18.17, 0.01: 23.46},
    (3, 0.10): {0.10: 13.69, 0.05: 15.72, 0.01: 20.30},
    (3, 0.15): {0.10: 12.49, 0.05: 14.30, 0.01: 18.35},
    (3, 0.20): {0.10: 11.69, 0.05: 13.41, 0.01: 17.05},
    # q=4
    (4, 0.05): {0.10: 17.71, 0.05: 20.08, 0.01: 25.32},
    (4, 0.10): {0.10: 15.41, 0.05: 17.44, 0.01: 22.12},
    (4, 0.15): {0.10: 14.11, 0.05: 15.98, 0.01: 19.99},
    (4, 0.20): {0.10: 13.30, 0.05: 14.98, 0.01: 18.60},
    # q=5
    (5, 0.05): {0.10: 19.49, 0.05: 21.84, 0.01: 27.14},
    (5, 0.10): {0.10: 17.07, 0.05: 19.11, 0.01: 23.72},
    (5, 0.15): {0.10: 15.63, 0.05: 17.54, 0.01: 21.55},
    (5, 0.20): {0.10: 14.77, 0.05: 16.49, 0.01: 20.16},
    # q=6
    (6, 0.05): {0.10: 21.15, 0.05: 23.54, 0.01: 28.84},
    (6, 0.10): {0.10: 18.63, 0.05: 20.68, 0.01: 25.28},
    (6, 0.15): {0.10: 17.15, 0.05: 18.97, 0.01: 23.06},
    (6, 0.20): {0.10: 16.15, 0.05: 17.94, 0.01: 21.51},
    # q=7
    (7, 0.05): {0.10: 22.75, 0.05: 25.17, 0.01: 30.45},
    (7, 0.10): {0.10: 20.15, 0.05: 22.23, 0.01: 26.80},
    (7, 0.15): {0.10: 18.55, 0.05: 20.43, 0.01: 24.50},
    (7, 0.20): {0.10: 17.49, 0.05: 19.29, 0.01: 22.97},
    # q=8
    (8, 0.05): {0.10: 24.28, 0.05: 26.69, 0.01: 32.04},
    (8, 0.10): {0.10: 21.58, 0.05: 23.68, 0.01: 28.27},
    (8, 0.15): {0.10: 19.91, 0.05: 21.84, 0.01: 25.84},
    (8, 0.20): {0.10: 18.81, 0.05: 20.57, 0.01: 24.40},
    # q=9
    (9, 0.05): {0.10: 25.73, 0.05: 28.18, 0.01: 33.53},
    (9, 0.10): {0.10: 22.99, 0.05: 25.08, 0.01: 29.70},
    (9, 0.15): {0.10: 21.23, 0.05: 23.17, 0.01: 27.22},
    (9, 0.20): {0.10: 20.06, 0.05: 21.88, 0.01: 25.70},
    # q=10
    (10, 0.05): {0.10: 27.18, 0.05: 29.60, 0.01: 34.97},
    (10, 0.10): {0.10: 24.34, 0.05: 26.45, 0.01: 31.11},
    (10, 0.15): {0.10: 22.53, 0.05: 24.44, 0.01: 28.54},
    (10, 0.20): {0.10: 21.28, 0.05: 23.10, 0.01: 26.90},
}

_EXPF_CRITICAL_VALUES: dict[tuple[int, float], dict[float, float]] = {
    # q=1
    (1, 0.05): {0.10: 3.15, 0.05: 4.39, 0.01: 7.08},
    (1, 0.10): {0.10: 2.72, 0.05: 3.78, 0.01: 6.13},
    (1, 0.15): {0.10: 2.51, 0.05: 3.42, 0.01: 5.48},
    (1, 0.20): {0.10: 2.36, 0.05: 3.16, 0.01: 4.88},
    # q=2
    (2, 0.05): {0.10: 4.39, 0.05: 5.67, 0.01: 8.47},
    (2, 0.10): {0.10: 3.76, 0.05: 4.87, 0.01: 7.34},
    (2, 0.15): {0.10: 3.40, 0.05: 4.38, 0.01: 6.48},
    (2, 0.20): {0.10: 3.16, 0.05: 4.03, 0.01: 5.82},
    # q=3
    (3, 0.05): {0.10: 5.45, 0.05: 6.78, 0.01: 9.62},
    (3, 0.10): {0.10: 4.71, 0.05: 5.82, 0.01: 8.30},
    (3, 0.15): {0.10: 4.23, 0.05: 5.23, 0.01: 7.32},
    (3, 0.20): {0.10: 3.92, 0.05: 4.80, 0.01: 6.67},
    # q=4
    (4, 0.05): {0.10: 6.44, 0.05: 7.78, 0.01: 10.69},
    (4, 0.10): {0.10: 5.57, 0.05: 6.74, 0.01: 9.23},
    (4, 0.15): {0.10: 5.03, 0.05: 6.03, 0.01: 8.14},
    (4, 0.20): {0.10: 4.65, 0.05: 5.55, 0.01: 7.38},
    # q=5
    (5, 0.05): {0.10: 7.35, 0.05: 8.73, 0.01: 11.68},
    (5, 0.10): {0.10: 6.40, 0.05: 7.57, 0.01: 10.09},
    (5, 0.15): {0.10: 5.77, 0.05: 6.82, 0.01: 8.99},
    (5, 0.20): {0.10: 5.33, 0.05: 6.28, 0.01: 8.10},
    # q=6
    (6, 0.05): {0.10: 8.22, 0.05: 9.61, 0.01: 12.60},
    (6, 0.10): {0.10: 7.17, 0.05: 8.37, 0.01: 10.92},
    (6, 0.15): {0.10: 6.48, 0.05: 7.54, 0.01: 9.79},
    (6, 0.20): {0.10: 5.98, 0.05: 6.97, 0.01: 8.83},
    # q=7
    (7, 0.05): {0.10: 9.07, 0.05: 10.47, 0.01: 13.49},
    (7, 0.10): {0.10: 7.93, 0.05: 9.14, 0.01: 11.73},
    (7, 0.15): {0.10: 7.16, 0.05: 8.25, 0.01: 10.51},
    (7, 0.20): {0.10: 6.62, 0.05: 7.64, 0.01: 9.55},
    # q=8
    (8, 0.05): {0.10: 9.88, 0.05: 11.31, 0.01: 14.33},
    (8, 0.10): {0.10: 8.66, 0.05: 9.88, 0.01: 12.49},
    (8, 0.15): {0.10: 7.84, 0.05: 8.93, 0.01: 11.26},
    (8, 0.20): {0.10: 7.23, 0.05: 8.27, 0.01: 10.24},
    # q=9
    (9, 0.05): {0.10: 10.67, 0.05: 12.11, 0.01: 15.16},
    (9, 0.10): {0.10: 9.37, 0.05: 10.60, 0.01: 13.24},
    (9, 0.15): {0.10: 8.49, 0.05: 9.60, 0.01: 11.96},
    (9, 0.20): {0.10: 7.84, 0.05: 8.88, 0.01: 10.95},
    # q=10
    (10, 0.05): {0.10: 11.44, 0.05: 12.90, 0.01: 15.94},
    (10, 0.10): {0.10: 10.06, 0.05: 11.32, 0.01: 13.96},
    (10, 0.15): {0.10: 9.14, 0.05: 10.24, 0.01: 12.60},
    (10, 0.20): {0.10: 8.43, 0.05: 9.49, 0.01: 11.62},
}

_AVEF_CRITICAL_VALUES: dict[tuple[int, float], dict[float, float]] = {
    # q=1
    (1, 0.05): {0.10: 4.12, 0.05: 5.47, 0.01: 8.48},
    (1, 0.10): {0.10: 3.58, 0.05: 4.71, 0.01: 7.28},
    (1, 0.15): {0.10: 3.26, 0.05: 4.26, 0.01: 6.47},
    (1, 0.20): {0.10: 3.04, 0.05: 3.91, 0.01: 5.77},
    # q=2
    (2, 0.05): {0.10: 5.55, 0.05: 6.92, 0.01: 10.01},
    (2, 0.10): {0.10: 4.76, 0.05: 5.95, 0.01: 8.63},
    (2, 0.15): {0.10: 4.30, 0.05: 5.34, 0.01: 7.62},
    (2, 0.20): {0.10: 3.98, 0.05: 4.90, 0.01: 6.84},
    # q=3
    (3, 0.05): {0.10: 6.82, 0.05: 8.22, 0.01: 11.35},
    (3, 0.10): {0.10: 5.86, 0.05: 7.05, 0.01: 9.77},
    (3, 0.15): {0.10: 5.28, 0.05: 6.34, 0.01: 8.65},
    (3, 0.20): {0.10: 4.87, 0.05: 5.80, 0.01: 7.82},
    # q=4
    (4, 0.05): {0.10: 7.98, 0.05: 9.43, 0.01: 12.59},
    (4, 0.10): {0.10: 6.89, 0.05: 8.10, 0.01: 10.83},
    (4, 0.15): {0.10: 6.21, 0.05: 7.27, 0.01: 9.59},
    (4, 0.20): {0.10: 5.72, 0.05: 6.65, 0.01: 8.73},
    # q=5
    (5, 0.05): {0.10: 9.10, 0.05: 10.57, 0.01: 13.78},
    (5, 0.10): {0.10: 7.85, 0.05: 9.11, 0.01: 11.86},
    (5, 0.15): {0.10: 7.07, 0.05: 8.18, 0.01: 10.50},
    (5, 0.20): {0.10: 6.50, 0.05: 7.49, 0.01: 9.55},
    # q=6
    (6, 0.05): {0.10: 10.17, 0.05: 11.67, 0.01: 14.91},
    (6, 0.10): {0.10: 8.78, 0.05: 10.06, 0.01: 12.84},
    (6, 0.15): {0.10: 7.91, 0.05: 9.04, 0.01: 11.38},
    (6, 0.20): {0.10: 7.28, 0.05: 8.28, 0.01: 10.35},
    # q=7
    (7, 0.05): {0.10: 11.21, 0.05: 12.74, 0.01: 15.99},
    (7, 0.10): {0.10: 9.69, 0.05: 10.98, 0.01: 13.78},
    (7, 0.15): {0.10: 8.73, 0.05: 9.87, 0.01: 12.25},
    (7, 0.20): {0.10: 8.02, 0.05: 9.06, 0.01: 11.14},
    # q=8
    (8, 0.05): {0.10: 12.21, 0.05: 13.77, 0.01: 17.05},
    (8, 0.10): {0.10: 10.56, 0.05: 11.88, 0.01: 14.68},
    (8, 0.15): {0.10: 9.52, 0.05: 10.68, 0.01: 13.06},
    (8, 0.20): {0.10: 8.74, 0.05: 9.78, 0.01: 11.91},
    # q=9
    (9, 0.05): {0.10: 13.19, 0.05: 14.77, 0.01: 18.03},
    (9, 0.10): {0.10: 11.43, 0.05: 12.76, 0.01: 15.58},
    (9, 0.15): {0.10: 10.28, 0.05: 11.48, 0.01: 13.88},
    (9, 0.20): {0.10: 9.45, 0.05: 10.51, 0.01: 12.65},
    # q=10
    (10, 0.05): {0.10: 14.13, 0.05: 15.74, 0.01: 18.99},
    (10, 0.10): {0.10: 12.26, 0.05: 13.62, 0.01: 16.43},
    (10, 0.15): {0.10: 11.03, 0.05: 12.24, 0.01: 14.68},
    (10, 0.20): {0.10: 10.14, 0.05: 11.23, 0.01: 13.40},
}


def _lookup_critical_values(
    table: dict[tuple[int, float], dict[float, float]],
    q: int,
    pi0: float,
) -> dict[float, float]:
    """Look up critical values for given q and trimming.

    If exact (q, pi0) is not in the table, uses the nearest available
    trimming for that q, or the largest available q if q > 10.

    Parameters
    ----------
    table : dict
        One of the three CV tables.
    q : int
        Number of breaking parameters.
    pi0 : float
        Trimming fraction.

    Returns
    -------
    dict[float, float]
        {significance_level: critical_value} for 0.10, 0.05, 0.01.
    """
    q_eff = min(max(q, 1), 10)

    available_pi0 = sorted({k[1] for k in table if k[0] == q_eff})
    if not available_pi0:
        return {0.10: np.nan, 0.05: np.nan, 0.01: np.nan}

    # Find nearest trimming
    best_pi0 = min(available_pi0, key=lambda p: abs(p - pi0))
    return table[(q_eff, best_pi0)]


def _coarse_pvalue(stat: float, critical_dict: dict[float, float]) -> float:
    """Compute a coarse p-value from critical value comparison.

    Parameters
    ----------
    stat : float
        Test statistic value.
    critical_dict : dict[float, float]
        {significance_level: critical_value}.

    Returns
    -------
    float
        Approximate p-value: < 0.01 returns 0.005, < 0.05 returns 0.025,
        < 0.10 returns 0.075, otherwise 0.20.
    """
    if np.isnan(stat):
        return np.nan
    cv_01 = critical_dict.get(0.01, np.inf)
    cv_05 = critical_dict.get(0.05, np.inf)
    cv_10 = critical_dict.get(0.10, np.inf)

    if stat >= cv_01:
        return 0.005
    elif stat >= cv_05:
        return 0.025
    elif stat >= cv_10:
        return 0.075
    else:
        return 0.20


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------


@dataclass
class AndrewsPlobergerResults(BreakTestResultsBase):
    """Results from the Andrews-Ploberger structural break test.

    Contains three test statistics (SupF, ExpF, AveF) computed from the
    same F-statistic sequence, along with critical values and approximate
    p-values.

    Attributes
    ----------
    test_name : str
        Name of the test ("Andrews-Ploberger").
    nobs : int
        Number of observations.
    n_breaks : int
        Number of breaks detected (0 or 1).
    break_indices : Sequence[int]
        Break index (argmax of F-sequence) if any statistic rejects.
    sup_f : float
        SupF statistic (maximum of F-sequence).
    exp_f : float
        ExpF statistic (log of average of exp(F/2)).
    ave_f : float
        AveF statistic (average of F-sequence).
    sup_f_critical : dict[float, float]
        Critical values for SupF at 10%, 5%, 1%.
    exp_f_critical : dict[float, float]
        Critical values for ExpF at 10%, 5%, 1%.
    ave_f_critical : dict[float, float]
        Critical values for AveF at 10%, 5%, 1%.
    sup_f_pvalue : float
        Approximate p-value for SupF.
    exp_f_pvalue : float
        Approximate p-value for ExpF.
    ave_f_pvalue : float
        Approximate p-value for AveF.
    f_sequence : NDArray[np.floating]
        F-statistic at each candidate break date.
    candidate_indices : NDArray[np.intp]
        Observation indices corresponding to each F-statistic.
    sup_f_break_index : int
        Estimated break date (argmax of F-sequence).
    trimming : float
        Trimming fraction used.
    q : int
        Number of breaking parameters.
    p : int
        Number of non-breaking parameters.
    significance_level : float
        Significance level used for rejection decisions.
    _test : AndrewsPlobergerTest | None
        Reference back to the test object (not shown in repr).
    """

    sup_f: float = np.nan
    exp_f: float = np.nan
    ave_f: float = np.nan
    sup_f_critical: dict[float, float] = field(default_factory=dict)
    exp_f_critical: dict[float, float] = field(default_factory=dict)
    ave_f_critical: dict[float, float] = field(default_factory=dict)
    sup_f_pvalue: float = np.nan
    exp_f_pvalue: float = np.nan
    ave_f_pvalue: float = np.nan
    f_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    candidate_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sup_f_break_index: int = 0
    trimming: float = 0.15
    q: int = 0
    p: int = 0
    significance_level: float = 0.05
    _test: AndrewsPlobergerTest | None = field(default=None, repr=False)

    @property
    def rejected(self) -> dict[str, bool]:
        """Whether H0 was rejected for each test statistic."""
        alpha = self.significance_level
        return {
            "SupF": self.sup_f_pvalue < alpha,
            "ExpF": self.exp_f_pvalue < alpha,
            "AveF": self.ave_f_pvalue < alpha,
        }

    def plot(
        self,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the F-statistic sequence.

        Convenience method delegating to
        :func:`regimes.visualization.andrews_ploberger.plot_f_sequence`.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axes to plot on.
        **kwargs
            Additional keyword arguments passed to ``plot_f_sequence``.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib figure and axes.
        """
        from regimes.visualization.andrews_ploberger import plot_f_sequence

        return plot_f_sequence(self, ax=ax, **kwargs)

    def summary(self) -> str:
        """Generate a text summary of Andrews-Ploberger test results.

        Returns
        -------
        str
            Formatted summary including all three statistics, critical
            values, p-values, and rejection decisions.
        """
        alpha = self.significance_level
        lines = []
        lines.append("=" * 78)
        lines.append(f"{'Andrews-Ploberger Structural Break Test':^78}")
        lines.append("=" * 78)
        lines.append(f"Number of observations:   {self.nobs:>10}")
        lines.append(f"Breaking parameters (q):  {self.q:>10}")
        lines.append(f"Non-breaking params (p):  {self.p:>10}")
        lines.append(f"Trimming:                 {self.trimming:>10.2f}")
        lines.append(f"Significance level:       {self.significance_level:>10.3f}")
        lines.append(f"Estimated break date:     {self.sup_f_break_index:>10}")
        lines.append("-" * 78)

        # Header row
        lines.append(
            f"\n{'Statistic':>12} {'Value':>12} "
            f"{'CV(10%)':>10} {'CV(5%)':>10} {'CV(1%)':>10} "
            f"{'p-value':>10} {'Reject':>8}"
        )
        lines.append("-" * 78)

        # SupF row
        cv10 = self.sup_f_critical.get(0.10, np.nan)
        cv05 = self.sup_f_critical.get(0.05, np.nan)
        cv01 = self.sup_f_critical.get(0.01, np.nan)
        reject_sup = "Yes" if self.sup_f_pvalue < alpha else "No"
        pv_str = _format_pvalue(self.sup_f_pvalue)
        lines.append(
            f"{'SupF':>12} {self.sup_f:>12.4f} "
            f"{cv10:>10.2f} {cv05:>10.2f} {cv01:>10.2f} "
            f"{pv_str:>10} {reject_sup:>8}"
        )

        # ExpF row
        cv10 = self.exp_f_critical.get(0.10, np.nan)
        cv05 = self.exp_f_critical.get(0.05, np.nan)
        cv01 = self.exp_f_critical.get(0.01, np.nan)
        reject_exp = "Yes" if self.exp_f_pvalue < alpha else "No"
        pv_str = _format_pvalue(self.exp_f_pvalue)
        lines.append(
            f"{'ExpF':>12} {self.exp_f:>12.4f} "
            f"{cv10:>10.2f} {cv05:>10.2f} {cv01:>10.2f} "
            f"{pv_str:>10} {reject_exp:>8}"
        )

        # AveF row
        cv10 = self.ave_f_critical.get(0.10, np.nan)
        cv05 = self.ave_f_critical.get(0.05, np.nan)
        cv01 = self.ave_f_critical.get(0.01, np.nan)
        reject_ave = "Yes" if self.ave_f_pvalue < alpha else "No"
        pv_str = _format_pvalue(self.ave_f_pvalue)
        lines.append(
            f"{'AveF':>12} {self.ave_f:>12.4f} "
            f"{cv10:>10.2f} {cv05:>10.2f} {cv01:>10.2f} "
            f"{pv_str:>10} {reject_ave:>8}"
        )

        lines.append("-" * 78)
        n_rejected = sum(1 for v in self.rejected.values() if v)
        lines.append(
            f"\nRejected H0 for {n_rejected} of 3 statistics "
            f"at the {alpha:.0%} level"
        )
        lines.append("=" * 78)
        return "\n".join(lines)


def _format_pvalue(pv: float) -> str:
    """Format a coarse p-value for display."""
    if np.isnan(pv):
        return "n/a"
    if pv <= 0.005:
        return "< 0.01"
    elif pv <= 0.025:
        return "< 0.05"
    elif pv <= 0.075:
        return "< 0.10"
    else:
        return "> 0.10"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class AndrewsPlobergerTest(BreakTestBase):
    """Andrews-Ploberger test for a single structural break at unknown date.

    Tests whether regression coefficients change at an unknown break point
    by computing F-statistics at every candidate break date in a trimmed
    range. Reports three summary statistics: SupF, ExpF, and AveF.

    Parameters
    ----------
    endog : ArrayLike
        Dependent variable (n_obs,).
    exog : ArrayLike | None
        Exogenous regressors whose coefficients do NOT break.
    exog_break : ArrayLike | None
        Regressors whose coefficients may break. If None, defaults to
        a constant (mean-shift model).

    Examples
    --------
    >>> import numpy as np
    >>> from regimes import AndrewsPlobergerTest
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    >>> test = AndrewsPlobergerTest(y)
    >>> results = test.fit()
    >>> print(results.summary())

    Notes
    -----
    The test assumes normally distributed, homoskedastic errors for
    the F-distribution to be exact. It is valid asymptotically under
    weaker assumptions.

    The trimming parameter pi0 controls how close to the sample
    boundaries candidate break dates can be. The default of 0.15
    (Andrews's recommendation) excludes the first and last 15% of
    observations.

    References
    ----------
    Andrews, D. W. K. (1993). Tests for parameter instability and structural
        change with unknown change point. Econometrica, 61(4), 821-856.

    Andrews, D. W. K. & Ploberger, W. (1994). Optimal tests when a nuisance
        parameter is present only under the alternative. Econometrica, 62(6),
        1383-1414.
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike | None = None,
        exog_break: ArrayLike | None = None,
    ) -> None:
        """Initialize the Andrews-Ploberger test."""
        if exog_break is None:
            exog_break = np.ones((len(np.asarray(endog)), 1))

        super().__init__(endog, exog, exog_break)

    @classmethod
    def from_model(
        cls,
        model: OLS | AR | ADL,
        break_vars: Literal["all", "const"] = "all",
    ) -> AndrewsPlobergerTest:
        """Create AndrewsPlobergerTest from an OLS, AR, or ADL model.

        Parameters
        ----------
        model : OLS | AR | ADL
            Model to test for structural breaks.
        break_vars : "all" | "const"
            Which variables can have breaks:
            - "all": All regressors can break (default)
            - "const": Only intercept can break (mean-shift model)

        Returns
        -------
        AndrewsPlobergerTest
            Test instance ready for .fit()

        Notes
        -----
        For AR and ADL models, the test uses the effective sample (after
        dropping initial observations for lags).
        """
        from regimes.models.adl import ADL as ADLModel
        from regimes.models.ar import AR as ARModel
        from regimes.models.ols import OLS as OLSModel

        if isinstance(model, (ARModel, ADLModel)):
            y, X, _ = model._build_design_matrix()
            endog = y
            exog_all = X
        elif isinstance(model, OLSModel):
            endog = model.endog
            exog_all = model.exog
        else:
            raise TypeError(
                f"model must be OLS, AR, or ADL, got {type(model).__name__}"
            )

        if exog_all is None:
            exog_all = np.ones((len(endog), 1))

        if break_vars == "all":
            return cls(endog, exog_break=exog_all)
        elif break_vars == "const":
            return cls(endog, exog=exog_all)
        else:
            raise ValueError(f"break_vars must be 'all' or 'const', got {break_vars!r}")

    @property
    def n_break_params(self) -> int:
        """Number of breaking regressors (q)."""
        if self.exog_break is None:
            return 0
        return self.exog_break.shape[1]

    @property
    def n_nonbreak_params(self) -> int:
        """Number of non-breaking regressors (p)."""
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    def _compute_ssr(
        self,
        y: NDArray[np.floating[Any]],
        X: NDArray[np.floating[Any]],
    ) -> float:
        """Compute sum of squared residuals from OLS.

        Parameters
        ----------
        y : NDArray[np.floating]
            Dependent variable.
        X : NDArray[np.floating]
            Regressor matrix.

        Returns
        -------
        float
            Sum of squared residuals.
        """
        try:
            beta, residuals, _rank, _s = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) > 0:
                return float(residuals[0])
            else:
                return float(np.sum((y - X @ beta) ** 2))
        except np.linalg.LinAlgError:
            return np.inf

    def _build_regressor_matrix(
        self,
        start: int,
        end: int,
    ) -> NDArray[np.floating[Any]]:
        """Build the full regressor matrix for a segment [start, end).

        Parameters
        ----------
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).

        Returns
        -------
        NDArray[np.floating]
            Regressor matrix for the segment.
        """
        parts = []
        if self.exog_break is not None:
            parts.append(self.exog_break[start:end])
        if self.exog is not None:
            parts.append(self.exog[start:end])

        if not parts:
            return np.ones((end - start, 1))

        return np.column_stack(parts)

    def _compute_f_statistic(self, tau: int) -> float:
        """Compute the F-statistic for a single candidate break date.

        F(τ) = [(SSR₀ - SSR₁ - SSR₂) / q] / [(SSR₁ + SSR₂) / (T - 2q - p)]

        Parameters
        ----------
        tau : int
            Candidate break date (index).

        Returns
        -------
        float
            F-statistic at this break date.
        """
        T = self.nobs
        q = self.n_break_params
        p = self.n_nonbreak_params

        # Full-sample regression
        X_full = self._build_regressor_matrix(0, T)
        ssr_full = self._compute_ssr(self.endog, X_full)

        # Compute unrestricted SSR
        if self.exog is not None and self.exog_break is not None:
            # Partial break: non-breaking regressors common across regimes
            X1_break = self.exog_break[:tau]
            X2_break = self.exog_break[tau:]
            X_nonbreak = self.exog

            Z1_break = np.zeros((T, q))
            Z1_break[:tau] = X1_break
            Z2_break = np.zeros((T, q))
            Z2_break[tau:] = X2_break

            X_unrestricted = np.column_stack([Z1_break, Z2_break, X_nonbreak])
            ssr_unrestricted = self._compute_ssr(self.endog, X_unrestricted)
        else:
            # All coefficients break
            X1 = self._build_regressor_matrix(0, tau)
            X2 = self._build_regressor_matrix(tau, T)
            ssr1 = self._compute_ssr(self.endog[:tau], X1)
            ssr2 = self._compute_ssr(self.endog[tau:], X2)
            ssr_unrestricted = ssr1 + ssr2

        df_denom = T - 2 * q - p
        if df_denom <= 0 or ssr_unrestricted <= 0:
            return 0.0

        f_stat = ((ssr_full - ssr_unrestricted) / q) / (ssr_unrestricted / df_denom)
        return max(0.0, f_stat)

    def _compute_f_sequence(
        self,
        trimming: float,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
        """Compute the F-statistic at every candidate break date.

        Parameters
        ----------
        trimming : float
            Fraction of observations to trim from each end.

        Returns
        -------
        tuple[NDArray[np.floating], NDArray[np.intp]]
            (f_sequence, candidate_indices)
        """
        T = self.nobs
        q = self.n_break_params
        p = self.n_nonbreak_params

        # Determine trimmed range
        min_obs = max(int(np.ceil(T * trimming)), q + p + 1)
        start = min_obs
        end = T - min_obs

        if start >= end:
            return np.array([]), np.array([], dtype=int)

        candidate_indices = np.arange(start, end + 1)

        # Precompute full-sample SSR
        X_full = self._build_regressor_matrix(0, T)
        ssr_full = self._compute_ssr(self.endog, X_full)

        f_sequence = np.zeros(len(candidate_indices))
        for i, tau in enumerate(candidate_indices):
            # Compute unrestricted SSR for this break date
            if self.exog is not None and self.exog_break is not None:
                X1_break = self.exog_break[:tau]
                X2_break = self.exog_break[tau:]
                X_nonbreak = self.exog

                Z1_break = np.zeros((T, q))
                Z1_break[:tau] = X1_break
                Z2_break = np.zeros((T, q))
                Z2_break[tau:] = X2_break

                X_unrestricted = np.column_stack([Z1_break, Z2_break, X_nonbreak])
                ssr_unrestricted = self._compute_ssr(self.endog, X_unrestricted)
            else:
                X1 = self._build_regressor_matrix(0, tau)
                X2 = self._build_regressor_matrix(tau, T)
                ssr1 = self._compute_ssr(self.endog[:tau], X1)
                ssr2 = self._compute_ssr(self.endog[tau:], X2)
                ssr_unrestricted = ssr1 + ssr2

            df_denom = T - 2 * q - p
            if df_denom <= 0 or ssr_unrestricted <= 0:
                f_sequence[i] = 0.0
            else:
                f_val = ((ssr_full - ssr_unrestricted) / q) / (
                    ssr_unrestricted / df_denom
                )
                f_sequence[i] = max(0.0, f_val)

        return f_sequence, candidate_indices

    def fit(
        self,
        trimming: float = 0.15,
        significance: float = 0.05,
        **kwargs: Any,
    ) -> AndrewsPlobergerResults:
        """Perform the Andrews-Ploberger test.

        Parameters
        ----------
        trimming : float
            Fraction of observations trimmed from each end. Default is
            0.15 (Andrews's recommendation).
        significance : float
            Significance level for rejection decisions. Default is 0.05.
        **kwargs
            Additional arguments (reserved for future use).

        Returns
        -------
        AndrewsPlobergerResults
            Results with SupF, ExpF, AveF statistics, critical values,
            and rejection decisions.

        Raises
        ------
        ValueError
            If trimming is not in (0, 0.5) or if there are too few
            candidate break dates after trimming.
        """
        if not 0 < trimming < 0.5:
            raise ValueError(f"trimming must be in (0, 0.5), got {trimming}")

        q = self.n_break_params
        p = self.n_nonbreak_params

        # Compute F-sequence
        f_sequence, candidate_indices = self._compute_f_sequence(trimming)

        T_star = len(f_sequence)
        if T_star == 0:
            raise ValueError(
                "No candidate break dates after trimming. "
                "Increase sample size or decrease trimming."
            )

        # Compute the three statistics
        sup_f = float(np.max(f_sequence))
        ave_f = float(np.mean(f_sequence))

        # ExpF: use log-sum-exp trick for numerical stability
        half_f = 0.5 * f_sequence
        max_half_f = np.max(half_f)
        exp_f = float(
            max_half_f + np.log(np.mean(np.exp(half_f - max_half_f)))
        )

        # Break location
        sup_f_break_index = int(candidate_indices[np.argmax(f_sequence)])

        # Look up critical values
        sup_f_cv = _lookup_critical_values(_SUPF_CRITICAL_VALUES, q, trimming)
        exp_f_cv = _lookup_critical_values(_EXPF_CRITICAL_VALUES, q, trimming)
        ave_f_cv = _lookup_critical_values(_AVEF_CRITICAL_VALUES, q, trimming)

        # Compute coarse p-values
        sup_f_pv = _coarse_pvalue(sup_f, sup_f_cv)
        exp_f_pv = _coarse_pvalue(exp_f, exp_f_cv)
        ave_f_pv = _coarse_pvalue(ave_f, ave_f_cv)

        # Determine break indices
        any_rejected = (
            sup_f_pv < significance
            or exp_f_pv < significance
            or ave_f_pv < significance
        )
        break_indices: list[int] = [sup_f_break_index] if any_rejected else []
        n_breaks = 1 if any_rejected else 0

        return AndrewsPlobergerResults(
            test_name="Andrews-Ploberger",
            nobs=self.nobs,
            n_breaks=n_breaks,
            break_indices=break_indices,
            sup_f=sup_f,
            exp_f=exp_f,
            ave_f=ave_f,
            sup_f_critical=sup_f_cv,
            exp_f_critical=exp_f_cv,
            ave_f_critical=ave_f_cv,
            sup_f_pvalue=sup_f_pv,
            exp_f_pvalue=exp_f_pv,
            ave_f_pvalue=ave_f_pv,
            f_sequence=f_sequence,
            candidate_indices=candidate_indices,
            sup_f_break_index=sup_f_break_index,
            trimming=trimming,
            q=q,
            p=p,
            significance_level=significance,
            _test=self,
        )
