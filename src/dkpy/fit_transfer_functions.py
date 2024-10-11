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
            Transfer matrix evaluated at each frequency.
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

    def __init__(self):
        pass

    def fit(
        self,
        omega: np.ndarray,
        D_omega: np.ndarray,
        order: int = 0,
        block_structure: Optional[np.ndarray] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        pass


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
    scipy.linalg.LinAlgError
        If the system's ``D`` matrix is singular.
    ValueError
        If the system's ``D`` matrix is nonsquare.
    """
    Di = scipy.linalg.inv(ss.D)
    Ai = ss.A - ss.B @ Di @ ss.C
    Bi = ss.B @ Di
    Ci = -Di @ ss.C
    ssi = control.StateSpace(Ai, Bi, Ci, Di, ss.dt)
    return ssi
