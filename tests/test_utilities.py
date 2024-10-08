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
