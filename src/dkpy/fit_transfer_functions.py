"""Classes for fitting transfer functions to magnitudes."""

__all__ = [
    "TransferFunctionFit",
    "TfFitSlicot",
]

import abc
from typing import Optional, Tuple

import control
import numpy as np


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
