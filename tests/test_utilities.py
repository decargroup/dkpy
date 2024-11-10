"""Test :mod:`utilities`."""

import control
import numpy as np
import pytest

import dkpy


class TestTfCloseCoeff:
    """Test :func:`_tf_close_coeff`."""

    @pytest.mark.parametrize(
        "tf",
        [
            control.TransferFunction([1, 2, 3], [4, 5, 6]),
            control.TransferFunction(
                [
                    [[2], [2, 0], [1]],
                    [[3, 0], [4], [2]],
                ],
                [
                    [[1, 2], [2, 3], [1, 0]],
                    [[3, 2], [3, 4], [2, 2]],
                ],
            ),
        ],
    )
    def test_equal(self, tf):
        """Test equal transfer functions."""
        assert dkpy._tf_close_coeff(tf, tf)

    @pytest.mark.parametrize(
        "tf_a, tf_b",
        [
            (
                control.TransferFunction([1, 2, 3], [4, 5, 6]),
                control.TransferFunction(
                    [
                        [[2], [2, 1], [1]],
                        [[3, 0], [4], [2]],
                    ],
                    [
                        [[1, 2], [2, 3], [1, 0]],
                        [[3, 2], [3, 4], [2, 2]],
                    ],
                ),
            ),
            (
                control.TransferFunction(
                    [
                        [[2], [2, 1], [1]],
                        [[3, 0], [4], [2]],
                    ],
                    [
                        [[1, 2], [2, 3], [1, 0]],
                        [[3, 2], [3, 4], [2, 2]],
                    ],
                ),
                control.TransferFunction(
                    [
                        [[2], [2, 0], [1]],
                        [[3, 0], [4], [2]],
                    ],
                    [
                        [[1, 2], [2, 3], [1, 0]],
                        [[3, 2], [3, 4], [2, 2]],
                    ],
                ),
            ),
            (
                control.TransferFunction(
                    [
                        [[2], [2, 0], [1]],
                        [[3, 0], [4], [2]],
                    ],
                    [
                        [[1, 2], [2, 3], [1, 0]],
                        [[3, 2], [2, 4], [2, 2]],
                    ],
                ),
                control.TransferFunction(
                    [
                        [[2], [2, 0], [1]],
                        [[3, 0], [4], [2]],
                    ],
                    [
                        [[1, 2], [2, 3], [1, 0]],
                        [[3, 2], [3, 4], [2, 2]],
                    ],
                ),
            ),
            (
                control.TransferFunction([1, 2, 3], [4, 5, 6], dt=0.1),
                control.TransferFunction([1, 2, 3], [4, 5, 6]),
            ),
        ],
    )
    def test_not_equal(self, tf_a, tf_b):
        """Test different transfer functions."""
        assert not dkpy._tf_close_coeff(tf_a, tf_b)


class TestEnsureTf:
    """Test :func:`_ensure_tf`."""

    @pytest.mark.parametrize(
        "arraylike_or_tf, dt, tf",
        [
            (
                control.TransferFunction([1], [1, 2, 3]),
                None,
                control.TransferFunction([1], [1, 2, 3]),
            ),
            (
                control.TransferFunction([1], [1, 2, 3]),
                0,
                control.TransferFunction([1], [1, 2, 3]),
            ),
            (
                2,
                None,
                control.TransferFunction([2], [1]),
            ),
            (
                np.array([2]),
                None,
                control.TransferFunction([2], [1]),
            ),
            (
                np.array([[2]]),
                None,
                control.TransferFunction([2], [1]),
            ),
            (
                np.array(
                    [
                        [2, 0, 3],
                        [1, 2, 3],
                    ]
                ),
                None,
                control.TransferFunction(
                    [
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            (
                np.array([2, 0, 3]),
                None,
                control.TransferFunction(
                    [
                        [[2], [0], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                    ],
                ),
            ),
        ],
    )
    def test_ensure(self, arraylike_or_tf, dt, tf):
        """Test nominal cases"""
        ensured_tf = dkpy._ensure_tf(arraylike_or_tf, dt)
        assert dkpy._tf_close_coeff(tf, ensured_tf)

    @pytest.mark.parametrize(
        "arraylike_or_tf, dt, exception",
        [
            (
                control.TransferFunction([1], [1, 2, 3]),
                0.1,
                ValueError,
            ),
            (
                control.TransferFunction([1], [1, 2, 3], 0.1),
                0,
                ValueError,
            ),
            (
                np.ones((1, 1, 1)),
                None,
                ValueError,
            ),
            (
                np.ones((1, 1, 1, 1)),
                None,
                ValueError,
            ),
        ],
    )
    def test_error_ensure(self, arraylike_or_tf, dt, exception):
        """Test error cases"""
        with pytest.raises(exception):
            dkpy._ensure_tf(arraylike_or_tf, dt)


class TestTfCombine:
    """Test :func:`_tf_combine`."""

    @pytest.mark.parametrize(
        "tf_array, tf",
        [
            # Continuous-time
            (
                [
                    [control.TransferFunction([1], [1, 1])],
                    [control.TransferFunction([2], [1, 0])],
                ],
                control.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                ),
            ),
            # Discrete-time
            (
                [
                    [control.TransferFunction([1], [1, 1], dt=1)],
                    [control.TransferFunction([2], [1, 0], dt=1)],
                ],
                control.TransferFunction(
                    [
                        [[1]],
                        [[2]],
                    ],
                    [
                        [[1, 1]],
                        [[1, 0]],
                    ],
                    dt=1,
                ),
            ),
            # Scalar
            (
                [
                    [2],
                    [control.TransferFunction([2], [1, 0])],
                ],
                control.TransferFunction(
                    [
                        [[2]],
                        [[2]],
                    ],
                    [
                        [[1]],
                        [[1, 0]],
                    ],
                ),
            ),
            # Matrix
            (
                [
                    [np.eye(3)],
                    [
                        control.TransferFunction(
                            [
                                [[2], [0], [3]],
                                [[1], [2], [3]],
                            ],
                            [
                                [[1], [1], [1]],
                                [[1], [1], [1]],
                            ],
                        )
                    ],
                ],
                control.TransferFunction(
                    [
                        [[1], [0], [0]],
                        [[0], [1], [0]],
                        [[0], [0], [1]],
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            # Inhomogeneous
            (
                [
                    [np.eye(3)],
                    [
                        control.TransferFunction(
                            [
                                [[2], [0]],
                                [[1], [2]],
                            ],
                            [
                                [[1], [1]],
                                [[1], [1]],
                            ],
                        ),
                        control.TransferFunction(
                            [
                                [[3]],
                                [[3]],
                            ],
                            [
                                [[1]],
                                [[1]],
                            ],
                        ),
                    ],
                ],
                control.TransferFunction(
                    [
                        [[1], [0], [0]],
                        [[0], [1], [0]],
                        [[0], [0], [1]],
                        [[2], [0], [3]],
                        [[1], [2], [3]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                ),
            ),
            # Discrete-time
            (
                [
                    [2],
                    [control.TransferFunction([2], [1, 0], dt=0.1)],
                ],
                control.TransferFunction(
                    [
                        [[2]],
                        [[2]],
                    ],
                    [
                        [[1]],
                        [[1, 0]],
                    ],
                    dt=0.1,
                ),
            ),
        ],
    )
    def test_combine(self, tf_array, tf):
        """Test combining transfer functions."""
        tf_combined = dkpy._tf_combine(tf_array)
        assert dkpy._tf_close_coeff(tf_combined, tf)

    @pytest.mark.parametrize(
        "tf_array, exception",
        [
            # Wrong timesteps
            (
                [
                    [control.TransferFunction([1], [1, 1], 0.1)],
                    [control.TransferFunction([2], [1, 0], 0.2)],
                ],
                ValueError,
            ),
            (
                [
                    [control.TransferFunction([1], [1, 1], 0.1)],
                    [control.TransferFunction([2], [1, 0], 0)],
                ],
                ValueError,
            ),
            # Too few dimensions
            (
                [
                    control.TransferFunction([1], [1, 1]),
                    control.TransferFunction([2], [1, 0]),
                ],
                ValueError,
            ),
            # Too many dimensions
            (
                [
                    [[control.TransferFunction([1], [1, 1], 0.1)]],
                    [[control.TransferFunction([2], [1, 0], 0)]],
                ],
                ValueError,
            ),
        ],
    )
    def test_error_combine(self, tf_array, exception):
        """Test error cases."""
        with pytest.raises(exception):
            dkpy._tf_combine(tf_array)


class TestTfEye:
    """Test :func:`_tf_eye`."""

    @pytest.mark.parametrize(
        "n, dt, tf_exp",
        [
            (
                1,
                None,
                control.TransferFunction([1], [1], dt=None),
            ),
            (
                2,
                0,
                control.TransferFunction(
                    [
                        [[1], [0]],
                        [[0], [1]],
                    ],
                    [
                        [[1], [1]],
                        [[1], [1]],
                    ],
                    dt=0,
                ),
            ),
            (
                2,
                1e-3,
                control.TransferFunction(
                    [
                        [[1], [0], [0]],
                        [[0], [1], [0]],
                        [[0], [0], [1]],
                    ],
                    [
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]],
                    ],
                    dt=1e-3,
                ),
            ),
        ],
    )
    def test_tf_eye(self, n, dt, tf_exp):
        """Test :func:`_tf_eye`."""
        tf = dkpy._tf_eye(n, dt)
        assert dkpy._tf_close_coeff(tf, tf)


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
        lmi_strictness = dkpy.utilities._auto_lmi_strictness(
            solver_params,
            scale,
        )
        np.testing.assert_allclose(lmi_strictness, lmi_strictness_exp)
