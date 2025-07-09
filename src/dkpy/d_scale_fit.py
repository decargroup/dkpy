"""Classes for fitting D-scale magnitudes."""

__all__ = [
    "DScaleFit",
    "DScaleFitSlicot",
]

import abc
from typing import Optional, Tuple, Union, List
import warnings

import control
import numpy as np
import scipy.linalg
import slycot

from . import utilities
from . import uncertainty_structure


class DScaleFit(metaclass=abc.ABCMeta):
    """D-scale fit base class."""

    @abc.abstractmethod
    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: Union[int, np.ndarray] = 0,
        uncertainty_structure: Optional[
            uncertainty_structure.UncertaintyBlockStructure
        ] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        """Fit D-scale magnitudes.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies (rad/s).
        D_omega : np.ndarray
            Transfer matrix evaluated at each frequency, with frequency as last
            dimension.
        order : Union[int, np.ndarray]
            Transfer function order to fit. Can be specified per-entry.
        uncertainty_structure : uncertainty_structure.UncertaintyBlockStructure
            Uncertainty block structure description.

        Returns
        -------
        Tuple[control.StateSpace, control.StateSpace]
            Fit state-space system and its inverse.

        Raises
        ------
        ValueError
            If ``order`` is an array but its dimensions are inconsistent with
            ``uncertainty_structure``.

        References
        ----------
        .. [#mussv] https://www.mathworks.com/help/robust/ref/mussv.html
        """
        raise NotImplementedError()


class DScaleFitSlicot(DScaleFit):
    """Fit D-scale magnitudes with SLICOT.

    Examples
    --------
    Compute ``mu`` and ``D`` at each frequency and fit a transfer matrix to ``D``

    >>> P, n_y, n_u, K = example_skogestad2006_p325
    >>> block_list = [
    ...     dkpy.ComplexFullBlock(1, 1),
    ...     dkpy.ComplexFullBlock(1, 1),
    ...     dkpy.ComplexFullBlock(2, 2),
    ... ]
    >>> uncertainty_structure = dkpy.UncertaintyBlockStructure(block_list)
    >>> omega = np.logspace(-3, 3, 61)
    >>> N = P.lft(K)
    >>> N_omega = N(1j * omega)
    >>> mu_omega, D_omega, info = dkpy.SsvLmiBisection().compute_ssv(
    ...     N_omega,
    ...     uncertainty_structure,
    ... )
    >>> D, D_inv = dkpy.DScaleFitSlicot().fit(omega, D_omega, 2, uncertainty_structure)
    """

    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: Union[int, np.ndarray] = 0,
        uncertainty_structure: Optional[
            uncertainty_structure.UncertaintyBlockStructure
        ] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        # Get mask
        if uncertainty_structure is None:
            mask = -1 * np.ones((D_omega.shape[0], D_omega.shape[1]), dtype=int)
        else:
            mask = uncertainty_structure.generate_d_scale_mask()
        # Check order dimensions
        orders = order if isinstance(order, np.ndarray) else order * np.ones_like(mask)
        if orders.shape != mask.shape:
            raise ValueError(
                "`order` must be an integer or an array whose dimensions are "
                "consistent with `uncertainty_structure`."
            )
        # Transfer matrix
        tf_array = np.zeros((D_omega.shape[0], D_omega.shape[1]), dtype=object)
        # Fit SISO transfer functions
        for row in range(D_omega.shape[0]):
            for col in range(D_omega.shape[1]):
                if mask[row, col] == 0:
                    tf_array[row, col] = control.TransferFunction([0], [1], dt=0)
                elif mask[row, col] == 1:
                    if isinstance(order, np.ndarray) and (orders[row, col] != 0):
                        warnings.warn(
                            "Entries of `order` in last uncertainty block "
                            "should be 0 since those transfer functions are "
                            "known to be 1. Ignoring value of "
                            f"`order[{row}, {col}]`."
                        )
                    tf_array[row, col] = control.TransferFunction([1], [1], dt=0)
                else:
                    n, A, B, C, D = slycot.sb10yd(
                        discfl=0,  # Continuous-time
                        flag=1,  # Constrain stable, minimum phase
                        lendat=omega.shape[0],
                        rfrdat=np.real(D_omega[row, col, :]),
                        ifrdat=np.imag(D_omega[row, col, :]),
                        omega=omega,
                        n=orders[row, col],
                        tol=0,  # Length of cache array
                    )
                    sys = control.StateSpace(A, B, C, D, dt=0)
                    tf_array[row, col] = control.ss2tf(sys)
        tf = utilities._tf_combine(tf_array)
        ss = control.tf2ss(tf)
        ss_inv = _invert_biproper_ss(ss)
        return ss, ss_inv


def _invert_biproper_ss(ss: control.StateSpace) -> control.StateSpace:
    """Invert a biproper, square state-space model.

    Parameters
    ----------
    ss : control.StateSpace
        Biproper state-space system.

    Returns
    -------
    control.StateSpace
        Inverted state-space system.

    Raises
    ------
    ValueError
        If the system's ``D`` matrix is singular.
    ValueError
        If the system's ``D`` matrix is nonsquare.
    """
    if ss.D.shape[0] != ss.D.shape[1]:
        raise ValueError("State-space `D` matrix is nonsquare.")
    try:
        Di = scipy.linalg.inv(ss.D)
    except scipy.linalg.LinAlgError:
        raise ValueError("State-space `D` matrix is singular.")
    Ai = ss.A - ss.B @ Di @ ss.C
    Bi = ss.B @ Di
    Ci = -Di @ ss.C
    ssi = control.StateSpace(Ai, Bi, Ci, Di, ss.dt)
    return ssi
