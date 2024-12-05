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
                dkpy.utilities._tf_eye(2),
                dkpy.utilities._tf_eye(2),
                2,
                1,
                dkpy.utilities._tf_eye(4),
                dkpy.utilities._tf_eye(3),
            ),
            (
                0.5 * dkpy.utilities._tf_eye(2),
                0.8 * dkpy.utilities._tf_eye(2),
                2,
                1,
                control.ss2tf(
                    control.append(
                        0.5 * dkpy.utilities._tf_eye(2),
                        dkpy.utilities._tf_eye(2),
                    )
                ),
                control.ss2tf(
                    control.append(
                        0.8 * dkpy.utilities._tf_eye(2),
                        dkpy.utilities._tf_eye(1),
                    )
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
