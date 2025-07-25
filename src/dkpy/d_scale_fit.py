"""Classes for fitting D-scale magnitudes."""

__all__ = [
    "DScaleFit",
    "DScaleFitSlicot",
]

import abc
from typing import Tuple, Union, List, Optional
import warnings

import control
import numpy as np
import scipy.linalg
import slycot

from . import uncertainty_structure


class DScaleFit(metaclass=abc.ABCMeta):
    """D-scale fit base class."""

    @abc.abstractmethod
    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: Union[int, np.ndarray] = 0,
        block_structure: Optional[
            Union[
                List[uncertainty_structure.UncertaintyBlock],
                List[List[int]],
                np.ndarray,
            ]
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
        block_structure : Optional[Union[List[uncertainty_structure.UncertaintyBlock], List[List[int], np.ndarray]]
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
    >>> block_structure = [
    ...     dkpy.ComplexFullBlock(1, 1),
    ...     dkpy.ComplexFullBlock(1, 1),
    ...     dkpy.ComplexFullBlock(2, 2),
    ... ]
    >>> omega = np.logspace(-3, 3, 61)
    >>> N = P.lft(K)
    >>> N_omega = N(1j * omega)
    >>> mu_omega, D_omega, info = dkpy.SsvLmiBisection().compute_ssv(
    ...     N_omega,
    ...     block_structure,
    ... )
    >>> D, D_inv = dkpy.DScaleFitSlicot().fit(omega, D_omega, 2, block_structure)
    """

    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: Union[int, np.ndarray] = 0,
        block_structure: Optional[
            Union[
                List[uncertainty_structure.UncertaintyBlock],
                List[List[int]],
                np.ndarray,
            ]
        ] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        # Get mask
        if block_structure is None:
            mask = -1 * np.ones((D_omega.shape[0], D_omega.shape[1]), dtype=int)
        else:
            block_structure = (
                uncertainty_structure._convert_block_structure_representation(
                    block_structure
                )
            )
            mask = _generate_d_scale_mask(block_structure)
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
        tf = control.combine_tf(tf_array)
        ss = control.tf2ss(tf)
        ss_inv = _invert_biproper_ss(ss)
        return ss, ss_inv


def _generate_d_scale_mask(
    block_structure: List[uncertainty_structure.UncertaintyBlock],
) -> np.ndarray:
    """Create a mask for the D-scale fit for the block structure.

    Entries known to be zero are set to 0. Entries known to be one are set to
    1. Entries to be fit numerically are set to -1.

    Parameters
    ----------
    block_structure : List[uncertainty_structure.UncertaintyBlock]
        Uncertainty block structure description.

    Returns
    -------
    np.ndarray
        Array of integers indicating zero, one, and unknown elements in the
        block structure.
    """
    num_blocks = len(block_structure)
    X_lst = []
    for i in range(num_blocks):
        # Uncertainty block
        block = block_structure[i]
        if not block.is_complex:
            raise NotImplementedError("Real perturbations are not yet supported.")
        if block.is_diagonal:
            raise NotImplementedError("Diagonal perturbations are not yet supported.")
        if not block.is_square:
            raise NotImplementedError("Nonsquare perturbations are not yet supported.")
        # Set last scaling to identity
        if i == num_blocks - 1:
            X_lst.append(np.eye(block.num_inputs, dtype=int))
        else:
            X_lst.append(-1 * np.eye(block.num_inputs, dtype=int))
    X = scipy.linalg.block_diag(*X_lst)
    return X


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
