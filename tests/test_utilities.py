"""Test :mod:`utilities`."""

import control
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
        ],
    )
    def test_combine(self, tf_array, tf):
        """Test combining transfer functions."""
        tf_combined = dkpy._tf_combine(tf_array)
        assert dkpy._tf_close_coeff(tf_combined, tf)
