"""Test :mod:`uncertainty_bound`."""

import control
import numpy as np

import dkpy
import dkpy.uncertainty_bound


def test_identify_uncertainty_bound():
    """Use the method on an example.

    Example from [MA00]_.
    """
    pnom = control.tf([2], [1, -2])
    p1 = pnom * control.tf([1], [0.06, 1])
    p2 = pnom * control.tf([-0.02, 1], [0.02, 1])
    p3 = pnom * control.tf([50**2], [1, 2 * 0.1 * 50, 50**2])
    freq_rng = np.logspace(-1, 3, 200)

    res_ub, w_E = dkpy.uncertainty_bound._identify_uncertainty_upper_bound(
        nom_model=pnom,
        off_nom_models=[p1, p2, p3],
        unc_str="mi",
        freq_rng=freq_rng,
        order=4,
    )

    assert res_ub is not None
    assert w_E is not None
