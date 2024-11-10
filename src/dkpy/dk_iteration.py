"""DK-iteration classes."""

__all__ = [
    "DkIteration",
]

import abc
from typing import Any, Dict, Tuple

import control
import numpy as np

from . import controller_synthesis, fit_transfer_functions, structured_singular_value


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


def _get_initial_d_scales(block_structure: np.ndarray) -> control.TransferFunction:
    """Generate initial identity D-scales based on block structure.

    Parameters
    ----------
    block_structure : np.ndarray
        2D array with 2 columns and as many rows as uncertainty blocks
        in Delta. The columns represent the number of rows and columns in
        each uncertainty block.

    Returns
    -------
    control.TransferFunction
        Identity D-scales.
    """
    pass


def _augment_d_scales(
    d_scales: control.TransferFunction,
    n_y: int,
    n_u: int,
) -> control.TransferFunction:
    """Augment D-scales with passthrough to account for outputs and inputs.

    Parameters
    ----------
    d_scales : control.TransferFunction
        D-scales.
    n_y : int
        Number of measurements (controller inputs).
    n_u : int
        Number of controller outputs.

    Returns
    -------
    control.TransferFunction
        Augmented D-scales.
    """
    pass
