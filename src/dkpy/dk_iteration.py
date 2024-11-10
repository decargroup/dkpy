"""DK-iteration classes."""

__all__ = [
    "DkIteration",
]

import abc
from typing import Any, Dict, Tuple, Union

import control
import numpy as np

from . import (
    controller_synthesis,
    fit_transfer_functions,
    structured_singular_value,
    utilities,
)


class DkIteration(metaclass=abc.ABCMeta):
    """DK-iteration base class."""

    def __init__(
        self,
        controller_synthesis: controller_synthesis.ControllerSynthesis,
        structured_singular_value: structured_singular_value.StructuredSingularValue,
        transfer_function_fit: fit_transfer_functions.TransferFunctionFit,
    ):
        """Instantiate :class:`DkIteration`.

        Parameters
        ----------
        controller_synthesis : dkpy.ControllerSynthesis
            A controller synthesis object.
        structured_singular_value : dkpy.StructuredSingularValue
            A structured singular value computation object.
        transfer_function_fit : dkpy.TransferFunctionFit
            A transfer function fit object.
        """
        self.controller_synthesis = controller_synthesis
        self.structured_singular_value = structured_singular_value
        self.transfer_function_fit = transfer_function_fit

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        """Synthesize controller.

        Parameters
        ----------
        P : control.StateSpace
            Generalized plant, with ``y`` and ``u`` as last outputs and inputs
            respectively.
        n_y : int
            Number of measurements (controller inputs).
        n_u : int
            Number of controller outputs.

        Returns
        -------
        Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]
            Controller, closed-loop system, structured singular value, solution
            information. If a controller cannot by synthesized, the first three
            elements of the tuple are ``None``, but solution information is
            still returned.
        """
        raise NotImplementedError()


def _get_initial_d_scales(block_structure: np.ndarray) -> control.StateSpace:
    """Generate initial identity D-scales based on block structure.

    Parameters
    ----------
    block_structure : np.ndarray
        2D array with 2 columns and as many rows as uncertainty blocks
        in Delta. The columns represent the number of rows and columns in
        each uncertainty block.

    Returns
    -------
    control.StateSpace
        Identity D-scales.
    """
    tf_lst = []
    for i in range(block_structure.shape[0]):
        if block_structure[i, 0] <= 0:
            raise NotImplementedError("Real perturbations are not yet supported.")
        if block_structure[i, 1] <= 0:
            raise NotImplementedError("Diagonal perturbations are not yet supported.")
        if block_structure[i, 0] != block_structure[i, 1]:
            raise NotImplementedError("Nonsquare perturbations are not yet supported.")
        tf_lst.append(utilities._tf_eye(block_structure[i, 0], dt=0))
    X = control.append(*tf_lst)
    return X


def _augment_d_scales(
    D: Union[control.TransferFunction, control.StateSpace],
    D_inv: Union[control.TransferFunction, control.StateSpace],
    n_y: int,
    n_u: int,
) -> Tuple[control.StateSpace, control.StateSpace]:
    """Augment D-scales with passthrough to account for outputs and inputs.

    Parameters
    ----------
    D : Union[control.TransferFunction, control.StateSpace]
        D-scales.
    D_inv : Union[control.TransferFunction, control.StateSpace]
        Inverse D-scales.
    n_y : int
        Number of measurements (controller inputs).
    n_u : int
        Number of controller outputs.

    Returns
    -------
    Tuple[control.StateSpace, control.StateSpace]
        Augmented D-scales and inverse D-scales.
    """
    D_aug = control.append(D, utilities._tf_eye(n_y))
    D_aug_inv = control.append(D_inv, utilities._tf_eye(n_u))
    return (D_aug, D_aug_inv)
