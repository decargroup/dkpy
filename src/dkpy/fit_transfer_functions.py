"""Classes for fitting transfer functions to magnitudes."""

__all__ = [
    "TransferFunctionFit",
    "TfFitSlicot",
]

import abc
from typing import Optional, Tuple

import control
import numpy as np
import scipy.linalg
import slycot

from . import utilities


class TransferFunctionFit(metaclass=abc.ABCMeta):
    """Transfer matrix fit base class."""

    @abc.abstractmethod
    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: int = 0,
        block_structure: Optional[np.ndarray] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        """Fit transfer matrix to magnitudes.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies (rad/s).
        D_omega : np.ndarray
            Transfer matrix evaluated at each frequency, with frequency as last
            dimension.
        order : int
            Transfer function order to fit.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        Tuple[control.StateSpace, control.StateSpace]
            Fit state-space system and its inverse.
        """
        raise NotImplementedError()


class TfFitSlicot(TransferFunctionFit):
    """Fit transfer matrix with SLICOT."""

    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: int = 0,
        block_structure: Optional[np.ndarray] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        # Get mask
        if block_structure is None:
            mask = np.ones((D_omega.shape[0], D_omega.shape[1]), dtype=bool)
        else:
            mask = _mask_from_block_structure(block_structure)
        # Transfer matrix
        tf_array = np.zeros((D_omega.shape[0], D_omega.shape[1]), dtype=object)
        # Fit SISO transfer functions
        for row in range(D_omega.shape[0]):
            for col in range(D_omega.shape[1]):
                if mask[row, col]:
                    n, A, B, C, D = slycot.sb10yd(
                        discfl=0,  # Continuous-time
                        flag=1,  # Constrain stable, minimum phase
                        lendat=omega.shape[0],
                        rfrdat=np.real(D_omega[row, col, :]),
                        ifrdat=np.imag(D_omega[row, col, :]),
                        omega=omega,
                        n=order,
                        tol=0,  # Length of cache array
                    )
                    sys = control.StateSpace(A, B, C, D)
                    tf_array[row, col] = control.ss2tf(sys)
                else:
                    tf_array[row, col] = control.TransferFunction([0], [1], dt=0)
        tf = utilities._tf_combine(tf_array)
        ss = control.tf2ss(tf)
        ss_inv = _invert_biproper_ss(ss)
        return ss, ss_inv


def _mask_from_block_structure(block_structure: np.ndarray) -> np.ndarray:
    """Create a binary mask from a specified block structure.

    Parameters
    ----------
    block_structure : np.ndarray
        2D array with 2 columns and as many rows as uncertainty blocks
        in Delta. The columns represent the number of rows and columns in
        each uncertainty block.

    Returns
    -------
    np.ndarray
        Array of booleans indicating nonzero elements in the block structure.
    """
    X_lst = []
    for i in range(block_structure.shape[0]):
        if block_structure[i, 0] <= 0:
            raise NotImplementedError("Real perturbations are not yet supported.")
        if block_structure[i, 1] <= 0:
            raise NotImplementedError("Diagonal perturbations are not yet supported.")
        if block_structure[i, 0] != block_structure[i, 1]:
            raise NotImplementedError("Nonsquare perturbations are not yet supported.")
        X_lst.append(np.eye(block_structure[i, 0], dtype=bool))
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
