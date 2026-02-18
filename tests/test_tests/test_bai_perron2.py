"""
Tests for Bai-Perron (1998, 2003) implementation correctness.

Verifies:
1. Partial structural change models (mixed breaking/non-breaking regressors).
2. Multicollinearity handling in model initialization.
3. Parameter mapping when converting to OLS representation.
"""

from __future__ import annotations

import numpy as np
import pytest
from regimes import BaiPerronTest, OLS

class TestBaiPerronCorrectness:
    """Verification of Bai-Perron estimation procedures."""

    def test_partial_structural_change_consistency(self) -> None:
        """
        Verify consistency of partial structural change estimation.
        
        Ensures that non-breaking regressors are constrained to have global 
        coefficients, while breaking regressors vary across regimes.
        Compares the estimated SSR against a restricted OLS benchmark.
        """
        np.random.seed(42)
        n = 100
        
        # DGP: y_t = 2*x_t + z_t * (1 if t < 50 else 3) + e_t
        # x is non-breaking, z is breaking (intercept)
        x = np.random.randn(n, 1)
        z = np.ones((n, 1))
        
        y = 2 * x.flatten()
        y[:50] += 1
        y[50:] += 3
        y += np.random.randn(n) * 0.1 
        
        bp = BaiPerronTest(y, exog=x, exog_break=z)
        results = bp.fit(max_breaks=1)
        
        # Benchmark: Restricted OLS
        # y = beta*x + delta1*D1 + delta2*D2
        X_restricted = np.zeros((n, 3))
        X_restricted[:, 0] = x.flatten()
        X_restricted[:50, 1] = 1 
        X_restricted[50:, 2] = 1 
        
        beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        resid_r = y - X_restricted @ beta_r
        ssr_restricted = np.sum(resid_r**2)
        
        # The iterative procedure should converge to the restricted OLS solution
        assert np.isclose(results.ssr[1], ssr_restricted, rtol=1e-3), \
            f"SSR divergence: BP={results.ssr[1]:.4f}, Restricted={ssr_restricted:.4f}"

    def test_initialization_collinearity(self) -> None:
        """
        Verify handling of constant terms in model initialization.
        
        When break_vars="const", the constant should be moved from the 
        non-breaking set (exog) to the breaking set (exog_break) to 
        prevent perfect multicollinearity.
        """
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 1 + x1 + np.random.randn(n)
        
        model = OLS(y, X, has_constant=False)
        bp = BaiPerronTest.from_model(model, break_vars="const")
        
        # Check partition of regressors
        has_const_in_exog = False
        if bp.exog is not None:
            has_const_in_exog = np.any(np.all(np.isclose(bp.exog, 1.0), axis=0))
        
        has_const_in_break = False
        if bp.exog_break is not None:
            has_const_in_break = np.any(np.all(np.isclose(bp.exog_break, 1.0), axis=0))
            
        assert has_const_in_break, "Constant missing from breaking regressors"
        assert not has_const_in_exog, "Constant retained in non-breaking regressors (collinearity risk)"

    def test_ols_parameter_mapping(self) -> None:
        """
        Verify parameter mapping in OLS conversion.
        
        Non-breaking regressors should map to a single global parameter.
        Breaking regressors should map to (m+1) regime-specific parameters.
        """
        np.random.seed(42)
        n = 100
        
        x_global = np.random.randn(n, 1)
        x_break = np.ones((n, 1))
        
        y = x_global.flatten()
        y[50:] += 5.0
        y += np.random.randn(n) * 0.1
        
        bp = BaiPerronTest(y, exog=x_global, exog_break=x_break)
        results = bp.fit(max_breaks=1)
        
        # Ensure we test the mapping logic even if detection varies
        if results.n_breaks == 0:
            results.n_breaks = 1
            results.break_indices = [50]
            
        ols = results.to_ols()
        
        # Expected: 1 global param + 2 regime intercepts = 3 parameters
        assert len(ols.params) == 3, \
            f"Incorrect parameter count: {len(ols.params)} (expected 3)"
            
        # Verify structure via naming convention
        # Global regressor should not have regime suffixes
        param_names = ols.param_names
        assert "x0_regime1" not in param_names, "Global regressor incorrectly treated as regime-dependent"
