"""D-K iteration classes."""

__all__ = [
    "DkIteration",
    "DkIterFixedOrder",
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
    """D-K iteration base class."""

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
        omega: np.ndarray,
        block_structure: np.ndarray,
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
        omega : np.ndarray
            Angular frequencies to evaluate D-scales (rad/s).
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]
            Controller, closed-loop system, structured singular value, solution
            information. If a controller cannot by synthesized, the first three
            elements of the tuple are ``None``, but solution information is
            still returned.
        """
        raise NotImplementedError()


class DkIterFixedOrder(DkIteration):
    """D-K iteration with a fixed number of iterations and fixed fit order."""

    def __init__(
        self,
        controller_synthesis: controller_synthesis.ControllerSynthesis,
        structured_singular_value: structured_singular_value.StructuredSingularValue,
        transfer_function_fit: fit_transfer_functions.TransferFunctionFit,
        n_iterations: int,
        fit_order: int,
    ):
        """Instantiate :class:`DkIterFixedOrder`.

        Parameters
        ----------
        controller_synthesis : dkpy.ControllerSynthesis
            A controller synthesis object.
        structured_singular_value : dkpy.StructuredSingularValue
            A structured singular value computation object.
        transfer_function_fit : dkpy.TransferFunctionFit
            A transfer function fit object.
        n_iterations : int
            Number of iterations.
        fit_order : int
            D-scale fit order.
        """
        self.controller_synthesis = controller_synthesis
        self.structured_singular_value = structured_singular_value
        self.transfer_function_fit = transfer_function_fit
        self.n_iterations = n_iterations
        self.fit_order = fit_order

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
        omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        # Solution information
        info = {}
        # Set up initial D-scales
        D = _get_initial_d_scales(block_structure)
        D_inv = _get_initial_d_scales(block_structure)
        D_aug, D_aug_inv = _augment_d_scales(D, D_inv, n_y=n_y, n_u=n_u)
        # Start iteration
        for i in range(self.n_iterations):
            # Synthesize controller
            K, _, gamma, info = self.controller_synthesis.synthesize(
                D_aug * P * D_aug_inv,
                n_y,
                n_u,
            )
            N = P.lft(K)
            # Compute structured singular values on grid
            N_omega = N(1j * omega)
            mus, Ds, info = self.structured_singular_value.compute_ssv(
                N_omega,
                block_structure=block_structure,
            )
            # Fit transfer functions to gridded D-scales
            D_fit, D_fit_inv = self.transfer_function_fit.fit(
                omega,
                Ds,
                order=self.fit_order,
                block_structure=block_structure,
            )
            # Augment D_scales with identity transfer functions
            D_aug, D_aug_inv = _augment_d_scales(
                D_fit,
                D_fit_inv,
                n_y=n_y,
                n_u=n_u,
            )
        # Synthesize controller one last time
        K, _, gamma, info = self.controller_synthesis.synthesize(
            D_aug * P * D_aug_inv,
            n_y,
            n_u,
        )
        N = P.lft(K)
        return (K, N, np.max(mus), info)


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
