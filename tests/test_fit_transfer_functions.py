"""Test :mod:`fit_transfer_functions`."""

import control
import numpy as np
import pytest

import dkpy


class TestTfFitSlicot:
    """Test :class:`TfFitSlicot`."""

    @pytest.mark.parametrize(
        "omega, tf, order, block_structure, atol",
        [
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction([10], [1]),
                0,
                None,
                1e-6,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction([1, 1], [1, 10]),
                1,
                None,
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 1],
                            [1, 1],
                        ],
                        [
                            [1, 1],
                            [1, 1],
                        ],
                    ],
                    [
                        [
                            [1, 10],
                            [1, 9],
                        ],
                        [
                            [1, 8],
                            [1, 10],
                        ],
                    ],
                ),
                1,
                None,
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 2, 1],
                            [1, 2, 1],
                        ],
                        [
                            [1, 2, 1],
                            [1, 2, 1],
                        ],
                    ],
                    [
                        [
                            [1, 10, 1],
                            [1, 9, 2],
                        ],
                        [
                            [1, 8, 3],
                            [1, 10, 4],
                        ],
                    ],
                ),
                2,
                None,
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 2, 1],
                            [1, 2, 1],
                        ],
                        [
                            [1, 2, 1],
                            [1, 2, 1],
                        ],
                    ],
                    [
                        [
                            [1, 10, 1],
                            [1, 9, 2],
                        ],
                        [
                            [1, 8, 3],
                            [1, 10, 4],
                        ],
                    ],
                ),
                2 * np.ones((2, 2)),
                None,
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 1],
                            [0],
                        ],
                        [
                            [0],
                            [1],
                        ],
                    ],
                    [
                        [
                            [1, 10],
                            [1],
                        ],
                        [
                            [1],
                            [1],
                        ],
                    ],
                ),
                1,
                np.array([[1, 1], [1, 1]]),
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 1],
                            [0],
                        ],
                        [
                            [0],
                            [1],
                        ],
                    ],
                    [
                        [
                            [1, 10],
                            [1],
                        ],
                        [
                            [1],
                            [1],
                        ],
                    ],
                ),
                np.diag([1, 1]),
                np.array([[1, 1], [1, 1]]),
                1e-2,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1],
                            [0],
                        ],
                        [
                            [0],
                            [1],
                        ],
                    ],
                    [
                        [
                            [1],
                            [1],
                        ],
                        [
                            [1],
                            [1],
                        ],
                    ],
                ),
                1,
                np.array([[2, 2]]),
                1e-2,
            ),
        ],
    )
    def test_tf_fit_slicot(self, omega, tf, order, block_structure, atol):
        """Test :class:`TfFitSlicot`."""
        D_omega = tf(1j * omega)
        if D_omega.ndim == 1:
            D_omega = D_omega.reshape((1, 1, -1))
        tf_fit, _ = dkpy.TfFitSlicot().fit(omega, D_omega, order, block_structure)
        D_omega_fit = tf_fit(1j * omega)
        if D_omega_fit.ndim == 1:
            D_omega_fit = D_omega_fit.reshape((1, 1, -1))
        np.testing.assert_allclose(D_omega, D_omega_fit, atol=atol)

    @pytest.mark.parametrize(
        "omega, tf, order, block_structure",
        [
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 1],
                            [1, 1],
                        ],
                        [
                            [1, 1],
                            [1, 1],
                        ],
                    ],
                    [
                        [
                            [1, 10],
                            [1, 10],
                        ],
                        [
                            [1, 10],
                            [1, 10],
                        ],
                    ],
                ),
                1,
                None,
            ),
            (
                np.logspace(-2, 2, 100),
                control.TransferFunction(
                    [
                        [
                            [1, 1],
                            [1, 1],
                            [1, 1],
                        ],
                        [
                            [1, 1],
                            [1, 1],
                            [1, 1],
                        ],
                    ],
                    [
                        [
                            [1, 10],
                            [1, 9],
                            [1, 7],
                        ],
                        [
                            [1, 8],
                            [1, 6],
                            [1, 10],
                        ],
                    ],
                ),
                1,
                None,
            ),
        ],
    )
    def test_tf_fit_slicot_error(self, omega, tf, order, block_structure):
        with pytest.raises(ValueError):
            D_omega = tf(1j * omega)
            if D_omega.ndim == 1:
                D_omega = D_omega.reshape((1, 1, -1))
            tf_fit, _ = dkpy.TfFitSlicot().fit(omega, D_omega, order, block_structure)


class TestMaskFromBlockStructure:
    """Test :func:`_mask_from_block_strucure`."""

    @pytest.mark.parametrize(
        "block_structure, mask_exp",
        [
            (
                np.array([[1, 1], [1, 1]]),
                np.array(
                    [
                        [-1, 0],
                        [0, 1],
                    ],
                    dtype=int,
                ),
            ),
            (
                np.array([[2, 2], [1, 1]]),
                np.array(
                    [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                    ],
                    dtype=int,
                ),
            ),
            (
                np.array([[1, 1], [2, 2]]),
                np.array(
                    [
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=int,
                ),
            ),
        ],
    )
    def test_mask_from_block_structure(self, block_structure, mask_exp):
        """Test :func:`_mask_from_block_strucure`."""
        mask = dkpy.fit_transfer_functions._mask_from_block_structure(block_structure)
        np.testing.assert_allclose(mask_exp, mask)


class TestInvertBiproperSs:
    """Test :func:`_invert_biproper_ss`."""

    @pytest.mark.parametrize(
        "tf",
        [
            control.TransferFunction([1, 2], [3, 4]),
            control.TransferFunction([1, 2, 1], [3, 6, 1]),
            control.TransferFunction([6, 2, 5], [5, 6, 7]),
        ],
    )
    def test_invert_biproper_ss(self, tf):
        """Test :func:`_invert_biproper_ss`."""
        ss = control.tf2ss(tf)
        ss_inv = dkpy.fit_transfer_functions._invert_biproper_ss(ss)
        eye_l = control.minreal(ss * ss_inv, verbose=False)
        eye_r = control.minreal(ss_inv * ss, verbose=False)
        omega = np.logspace(-3, 3, 100)
        for eye in [eye_l, eye_r]:
            mag, _, _ = eye.frequency_response(omega)
            np.testing.assert_allclose(1, eye.dcgain())
            np.testing.assert_allclose(np.ones_like(mag), mag)
