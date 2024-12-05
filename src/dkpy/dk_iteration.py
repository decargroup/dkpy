"""D-K iteration classes."""

__all__ = [
    "DScaleFitInfo",
    "DkIteration",
    "DkIterFixedOrder",
    "DkIterListOrder",
    "DkIterOrderCallback",
]

import abc
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import control
import numpy as np
import scipy.linalg

from . import (
    controller_synthesis,
    fit_transfer_functions,
    structured_singular_value,
    utilities,
)


class DScaleFitInfo:
    """Information about the D-scale fit accuracy."""

    def __init__(
        self,
        omega: np.ndarray,
        mu_omega: np.ndarray,
        D_omega: np.ndarray,
        mu_fit_omega: np.ndarray,
        D_fit_omega: np.ndarray,
        D_fit: control.StateSpace,
        block_structure: np.ndarray,
    ):
        """Instantiate :class:`DScaleFitInfo`.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies to evaluate D-scales (rad/s).
        mu_omega : np.ndarray
            Numerically computed structured singular value at each frequency.
        D_omega : np.ndarray
            Numerically computed D-scale magnitude at each frequency.
        mu_fit_omega : np.ndarray
            Fit structured singular value at each frequency.
        D_fit_omega : np.ndarray
            Fit D-scale magnitude at each frequency.
        D_fit : control.StateSpace
            Fit D-scale state-space representation.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.
        """
        self.omega = omega
        self.mu_omega = mu_omega
        self.D_omega = D_omega
        self.mu_fit_omega = mu_fit_omega
        self.D_fit_omega = D_fit_omega
        self.D_fit = D_fit
        self.block_structure = block_structure

    @classmethod
    def create_from_fit(
        cls,
        omega: np.ndarray,
        mu_omega: np.ndarray,
        D_omega: np.ndarray,
        P: control.StateSpace,
        K: control.StateSpace,
        D_fit: control.StateSpace,
        D_fit_inv: control.StateSpace,
        block_structure: np.ndarray,
    ) -> "DScaleFitInfo":
        """Instantiate :class:`DScaleFitInfo` from fit D-scales.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies to evaluate D-scales (rad/s).
        mu_omega : np.ndarray
            Numerically computed structured singular value at each frequency.
        D_omega : np.ndarray
            Numerically computed D-scale magnitude at each frequency.
        P : control.StateSpace
            Generalized plant.
        K : control.StateSpace
            Controller.
        D_fit : control.StateSpace
            Fit D-scale magnitude at each frequency.
        D_fit_inv : control.StateSpace
            Fit inverse D-scale magnitude at each frequency.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        DScaleFitInfo
            Instance of :class:`DScaleFitInfo`
        """
        # Compute ``mu(omega)`` based on fit D-scales
        N = P.lft(K)
        scaled_cl = (D_fit * N * D_fit_inv)(1j * omega)
        mu_fit_omega = np.array(
            [
                np.max(scipy.linalg.svdvals(scaled_cl[:, :, i]))
                for i in range(scaled_cl.shape[2])
            ]
        )
        # Compute ``D(omega)`` based on fit D-scales
        D_fit_omega = D_fit(1j * omega)
        return cls(
            omega,
            mu_omega,
            D_omega,
            mu_fit_omega,
            D_fit_omega,
            D_fit,
            block_structure,
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
    ) -> Tuple[
        control.StateSpace,
        control.StateSpace,
        float,
        List[DScaleFitInfo],
        Dict[str, Any],
    ]:
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
        Tuple[control.StateSpace, control.StateSpace, float, List[DScaleFitInfo], Dict[str, Any]]
            Controller, closed-loop system, structured singular value, D-scale
            fit info for each iteration, and solution information. If a
            controller cannot by synthesized, the first three elements of the
            tuple are ``None``, but fit and solution information are still
            returned.
        """
        raise NotImplementedError()

    def _get_fit_order(
        self,
        i: int,
        omega: np.ndarray,
        mu_omega: np.ndarray,
        D_omega: np.ndarray,
        P: control.StateSpace,
        K: control.StateSpace,
        block_structure: np.ndarray,
    ) -> Optional[Union[int, np.ndarray]]:
        """Get D-scale fit order.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies to evaluate D-scales (rad/s).
        mu_omega : np.ndarray
            Numerically computed structured singular value at each frequency.
        D_omega : np.ndarray
            Numerically computed D-scale magnitude at each frequency.
        P : control.StateSpace
            Generalized plant.
        K : control.StateSpace
            Controller.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        Optional[Union[int, np.ndarray]]
            D-scale fit order. If ``None``, iteration ends.
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
        fit_order: Union[int, np.ndarray],
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
        fit_order : Union[int, np.ndarray]
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
    ) -> Tuple[
        control.StateSpace,
        control.StateSpace,
        float,
        List[DScaleFitInfo],
        Dict[str, Any],
    ]:
        # Solution information
        info = {}
        d_scale_fit_info = []
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
            mu_omega, D_omega, info = self.structured_singular_value.compute_ssv(
                N_omega,
                block_structure=block_structure,
            )
            # Fit transfer functions to gridded D-scales
            D_fit, D_fit_inv = self.transfer_function_fit.fit(
                omega,
                D_omega,
                order=self.fit_order,
                block_structure=block_structure,
            )
            # Add D-scale fit info
            d_scale_fit_info.append(
                DScaleFitInfo.create_from_fit(
                    omega,
                    mu_omega,
                    D_omega,
                    P,
                    K,
                    D_fit,
                    D_fit_inv,
                    block_structure,
                )
            )
            # Augment D-scales with identity transfer functions
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
        return (K, N, np.max(mu_omega), d_scale_fit_info, info)


class DkIterListOrder(DkIteration):
    """D-K iteration with a fixed list of fit orders."""

    def __init__(
        self,
        controller_synthesis: controller_synthesis.ControllerSynthesis,
        structured_singular_value: structured_singular_value.StructuredSingularValue,
        transfer_function_fit: fit_transfer_functions.TransferFunctionFit,
        fit_orders: List[Union[int, np.ndarray]],
    ):
        """Instantiate :class:`DkIterListOrder`.

        Parameters
        ----------
        controller_synthesis : dkpy.ControllerSynthesis
            A controller synthesis object.
        structured_singular_value : dkpy.StructuredSingularValue
            A structured singular value computation object.
        transfer_function_fit : dkpy.TransferFunctionFit
            A transfer function fit object.
        fit_order : List[Union[int, np.ndarray]]
            D-scale fit orders.
        """
        self.controller_synthesis = controller_synthesis
        self.structured_singular_value = structured_singular_value
        self.transfer_function_fit = transfer_function_fit
        self.fit_orders = fit_orders

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
        omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[
        control.StateSpace,
        control.StateSpace,
        float,
        List[DScaleFitInfo],
        Dict[str, Any],
    ]:
        # Solution information
        info = {}
        d_scale_fit_info = []
        # Set up initial D-scales
        D = _get_initial_d_scales(block_structure)
        D_inv = _get_initial_d_scales(block_structure)
        D_aug, D_aug_inv = _augment_d_scales(D, D_inv, n_y=n_y, n_u=n_u)
        # Start iteration
        for fit_order in self.fit_orders:
            # Synthesize controller
            K, _, gamma, info = self.controller_synthesis.synthesize(
                D_aug * P * D_aug_inv,
                n_y,
                n_u,
            )
            N = P.lft(K)
            # Compute structured singular values on grid
            N_omega = N(1j * omega)
            mu_omega, D_omega, info = self.structured_singular_value.compute_ssv(
                N_omega,
                block_structure=block_structure,
            )
            # Fit transfer functions to gridded D-scales
            D_fit, D_fit_inv = self.transfer_function_fit.fit(
                omega,
                D_omega,
                order=fit_order,
                block_structure=block_structure,
            )
            # Add D-scale fit info
            d_scale_fit_info.append(
                DScaleFitInfo.create_from_fit(
                    omega,
                    mu_omega,
                    D_omega,
                    P,
                    K,
                    D_fit,
                    D_fit_inv,
                    block_structure,
                )
            )
            # Augment D-scales with identity transfer functions
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
        return (K, N, np.max(mu_omega), d_scale_fit_info, info)


class DkIterOrderCallback(DkIteration):
    """D-K iteration with a callback to select fit orders."""

    def __init__(
        self,
        controller_synthesis: controller_synthesis.ControllerSynthesis,
        structured_singular_value: structured_singular_value.StructuredSingularValue,
        transfer_function_fit: fit_transfer_functions.TransferFunctionFit,
        fit_order_callback: Optional[Callable] = None,
    ):
        """Instantiate :class:`DkIterListOrder`.

        Parameters
        ----------
        controller_synthesis : dkpy.ControllerSynthesis
            A controller synthesis object.
        structured_singular_value : dkpy.StructuredSingularValue
            A structured singular value computation object.
        transfer_function_fit : dkpy.TransferFunctionFit
            A transfer function fit object.
        fit_order_callback : Optional[Callable]
            Callback to select D-scale fit orders.
        """
        self.controller_synthesis = controller_synthesis
        self.structured_singular_value = structured_singular_value
        self.transfer_function_fit = transfer_function_fit
        self.fit_order_callback = fit_order_callback

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
        omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[
        control.StateSpace,
        control.StateSpace,
        float,
        List[DScaleFitInfo],
        Dict[str, Any],
    ]:
        # Solution information
        info = {}
        d_scale_fit_info = []
        # Set up initial D-scales
        D = _get_initial_d_scales(block_structure)
        D_inv = _get_initial_d_scales(block_structure)
        D_aug, D_aug_inv = _augment_d_scales(D, D_inv, n_y=n_y, n_u=n_u)
        iteration = 0
        done = False
        while not done:
            # Synthesize controller
            K, _, gamma, info = self.controller_synthesis.synthesize(
                D_aug * P * D_aug_inv,
                n_y,
                n_u,
            )
            N = P.lft(K)
            # Compute structured singular values on grid
            N_omega = N(1j * omega)
            mu_omega, D_omega, info = self.structured_singular_value.compute_ssv(
                N_omega,
                block_structure=block_structure,
            )
            fit_order, done = self.fit_order_callback(
                self,
                iteration,
                omega,
                mu_omega,
                D_omega,
                P,
                K,
                block_structure,
            )
            # Fit transfer functions to gridded D-scales
            D_fit, D_fit_inv = self.transfer_function_fit.fit(
                omega,
                D_omega,
                order=fit_order,
                block_structure=block_structure,
            )
            # Add D-scale fit info
            d_scale_fit_info.append(
                DScaleFitInfo.create_from_fit(
                    omega,
                    mu_omega,
                    D_omega,
                    P,
                    K,
                    D_fit,
                    D_fit_inv,
                    block_structure,
                )
            )
            # Augment D-scales with identity transfer functions
            D_aug, D_aug_inv = _augment_d_scales(
                D_fit,
                D_fit_inv,
                n_y=n_y,
                n_u=n_u,
            )
            # Increment iteration
            iteration += 1
        # Synthesize controller one last time
        K, _, gamma, info = self.controller_synthesis.synthesize(
            D_aug * P * D_aug_inv,
            n_y,
            n_u,
        )
        N = P.lft(K)
        return (K, N, np.max(mu_omega), d_scale_fit_info, info)


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
