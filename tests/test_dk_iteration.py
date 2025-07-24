"""Test :mod:`dk_iteration`."""

import control
import numpy as np
import pytest

import dkpy


class TestAugmentDScales:
    """Test :func:`_augment_d_scales`."""

    @pytest.mark.parametrize(
        "D, D_inv, n_y, n_u, D_aug_exp, D_aug_inv_exp",
        [
            (
                control.TransferFunction([1], [1]) * np.eye(2),
                control.TransferFunction([1], [1]) * np.eye(2),
                2,
                1,
                control.TransferFunction([1], [1]) * np.eye(4),
                control.TransferFunction([1], [1]) * np.eye(3),
            ),
            (
                control.TransferFunction([0.5], [1]) * np.eye(2),
                control.TransferFunction([0.8], [1]) * np.eye(2),
                2,
                1,
                control.append(
                    control.TransferFunction([0.5], [1]) * np.eye(2),
                    control.TransferFunction([1], [1]) * np.eye(2),
                ),
                control.append(
                    control.TransferFunction([0.8], [1]) * np.eye(2),
                    control.TransferFunction([1], [1]) * np.eye(1),
                ),
            ),
        ],
    )
    def test_augment_d_scales(self, D, D_inv, n_y, n_u, D_aug_exp, D_aug_inv_exp):
        """Test :func:`_augment_d_scales`."""
        D_aug, D_aug_inv = dkpy.dk_iteration._augment_d_scales(D, D_inv, n_y, n_u)
        D_aug_tf = control.ss2tf(D_aug)
        D_aug_inv_tf = control.ss2tf(D_aug_inv)
        assert dkpy.utilities._tf_close_coeff(D_aug_tf, D_aug_exp)
        assert dkpy.utilities._tf_close_coeff(D_aug_inv_tf, D_aug_inv_exp)
