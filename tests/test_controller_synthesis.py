"""Test :mod:`controller_synthesis`."""

import numpy as np
import pytest

import dkpy


class TestControllerSynthesis:
    """Compare :class:`HinfSynSlicot` and :class:`HinfSynLmi` solutions."""

    pass


class TestAutoLmiStrictness:
    """Test :func:`_auto_lmi_strictness`."""

    @pytest.mark.parametrize(
        "solver_params, scale, lmi_strictness_exp",
        [
            (
                {
                    "solver": "CLARABEL",
                },
                10,
                1e-7,
            ),
            (
                {
                    "solver": "CLARABEL",
                    "tol_feas": 1e-6,
                },
                10,
                1e-5,
            ),
            (
                {
                    "solver": "COPT",
                    "FeasTol": 1e-6,
                },
                10,
                1e-5,
            ),
            (
                {
                    "solver": "MOSEK",
                },
                10,
                1e-7,
            ),
            (
                {
                    "solver": "MOSEK",
                    "eps": 1e-3,
                },
                10,
                1e-2,
            ),
            (
                {
                    "solver": "MOSEK",
                    "mosek_params": {
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-4,
                    },
                },
                10,
                1e-3,
            ),
            # ``mosek_params`` takes precedence over ``eps``.
            (
                {
                    "solver": "MOSEK",
                    "eps": 1e-9,
                    "mosek_params": {
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-4,
                    },
                },
                10,
                1e-3,
            ),
            (
                {
                    "solver": "CVXOPT",
                    "abstol": 1e-6,
                },
                10,
                1e-5,
            ),
            (
                {
                    "solver": "SDPA",
                    "epsilonStar": 1e-6,
                },
                10,
                1e-5,
            ),
            (
                {
                    "solver": "SCS",
                    "eps": 1e-6,
                },
                10,
                1e-5,
            ),
        ],
    )
    def test_auto_lmi_strictness(self, solver_params, scale, lmi_strictness_exp):
        """Test :func:`_auto_lmi_strictness`."""
        lmi_strictness = dkpy.controller_synthesis._auto_lmi_strictness(
            solver_params,
            scale,
        )
        np.testing.assert_allclose(lmi_strictness, lmi_strictness_exp)
