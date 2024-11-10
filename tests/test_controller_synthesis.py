"""Test :mod:`controller_synthesis`."""

import control
import numpy as np

import dkpy


class TestControllerSynthesis:
    """Compare :class:`HinfSynSlicot` and :class:`HinfSynLmi` solutions.

    Based on Example 7 from [SGC97]_.
    """

    def test_compare_gamma(self):
        """Compare :class:`HinfSynSlicot` and :class:`HinfSynLmi` solutions."""
        # Process model
        A = np.array([[0, 10, 2], [-1, 1, 0], [0, 2, -5]])
        B1 = np.array([[1], [0], [1]])
        B2 = np.array([[0], [1], [0]])
        # Plant output
        C2 = np.array([[0, 1, 0]])
        D21 = np.array([[2]])
        D22 = np.array([[0]])
        # Hinf performance
        C1 = np.array([[1, 0, 0], [0, 0, 0]])
        D11 = np.array([[0], [0]])
        D12 = np.array([[0], [1]])
        # Dimensions
        n_y = 1
        n_u = 1
        # Create generalized plant
        B_gp = np.block([B1, B2])
        C_gp = np.block([[C1], [C2]])
        D_gp = np.block([[D11, D12], [D21, D22]])
        P = control.StateSpace(A, B_gp, C_gp, D_gp)
        # Compare controllers
        _, _, gamma_exp, _ = control.hinfsyn(P, n_y, n_u)
        _, _, gamma_slicot, _ = dkpy.HinfSynSlicot().synthesize(P, n_y, n_u)
        _, _, gamma_lmi, _ = dkpy.HinfSynLmi().synthesize(P, n_y, n_u)
        _, _, gamma_bisect, _ = dkpy.HinfSynLmiBisection().synthesize(P, n_y, n_u)
        np.testing.assert_allclose(gamma_slicot, gamma_exp)
        np.testing.assert_allclose(gamma_lmi, gamma_exp, atol=1e-4)
        np.testing.assert_allclose(gamma_bisect, gamma_exp, atol=1e-3)
