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
        D_l_omega: np.ndarray,
        D_r_omega: np.ndarray,
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
        D_l_omega : np.ndarray
            Transfer matrix evaluated at each frequency, with frequency as last
            dimension.
        D_r_omega: np.ndarray,
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
    >>> mu_omega, D_l_omega, D_r_omega, info = dkpy.SsvLmiBisection().compute_ssv(
    ...     N_omega,
    ...     block_structure,
    ... )
    >>> D, D_inv = dkpy.DScaleFitSlicot().fit(omega, D_l_omega, D_r_omega, 2, block_structure)
    """

    def fit(
        self,
        omega: np.ndarray,
        D_l_omega: np.ndarray,
        D_r_omega: np.ndarray,
        order: Union[int, List[int], np.ndarray] = 0,
        block_structure: Optional[
            Union[
                List[uncertainty_structure.UncertaintyBlock],
                List[List[int]],
                np.ndarray,
            ]
        ] = None,
    ) -> Tuple[control.StateSpace, control.StateSpace]:
        # Check D-scale dimensions
        if D_l_omega.shape[0] != D_l_omega.shape[1]:
            raise ValueError(
                "The left D-scale must be square "
                "(D_l_omega.shape[0] = D_l_omega.shape[1])."
            )
        if D_r_omega.shape[0] != D_r_omega.shape[1]:
            raise ValueError(
                "The right D-scale must be square "
                "(D_r_omega.shape[0] = D_r_omega.shape[1])."
            )
        # Get mask
        if block_structure is None:
            block_structure = [
                uncertainty_structure.ComplexFullBlock(
                    D_r_omega.shape[0], D_l_omega.shape[0]
                )
            ]
        else:
            block_structure = (
                uncertainty_structure._convert_block_structure_representation(
                    block_structure
                )
            )
        mask_l, mask_r = _generate_d_scale_mask(block_structure)
        if isinstance(order, int):
            orders_l = order * np.abs(mask_l)
            orders_r = order * np.abs(mask_r)
        else:
            order = np.array(order)
            if order.size != len(block_structure):
                raise ValueError(
                    "Length of `order` must be the same as length of `block_structure`."
                )
            orders_l_lst = [
                order[i]
                * np.ones(
                    (
                        block_structure[i].num_perf_outputs,
                        block_structure[i].num_perf_outputs,
                    )
                )
                for i in range(len(block_structure))
            ]
            orders_r_lst = [
                order[i]
                * np.ones(
                    (
                        block_structure[i].num_exog_inputs,
                        block_structure[i].num_exog_inputs,
                    )
                )
                for i in range(len(block_structure))
            ]
            orders_l = scipy.linalg.block_diag(*orders_l_lst)
            orders_r = scipy.linalg.block_diag(*orders_r_lst)

        # Transfer matrix
        tf_l_array = np.zeros((D_l_omega.shape[0], D_l_omega.shape[1]), dtype=object)
        tf_r_array = np.zeros((D_r_omega.shape[0], D_r_omega.shape[1]), dtype=object)
        # Fit SISO transfer functions of left scale
        for row in range(D_l_omega.shape[0]):
            for col in range(D_l_omega.shape[1]):
                if mask_l[row, col] == 0:
                    tf_l_array[row, col] = control.TransferFunction([0], [1], dt=0)
                elif mask_l[row, col] == 1:
                    if isinstance(order, np.ndarray) and (orders_l[row, col] != 0):
                        warnings.warn(
                            "Entries of `order` in last uncertainty block "
                            "should be 0 since those transfer functions are "
                            "known to be 1. Ignoring value of "
                            f"`order[{row}, {col}]`."
                        )
                    tf_l_array[row, col] = control.TransferFunction([1], [1], dt=0)
                else:
                    n, A, B, C, D = slycot.sb10yd(
                        discfl=0,  # Continuous-time
                        flag=1,  # Constrain stable, minimum phase
                        lendat=omega.shape[0],
                        rfrdat=np.real(D_l_omega[row, col, :]),
                        ifrdat=np.imag(D_l_omega[row, col, :]),
                        omega=omega,
                        n=orders_l[row, col],
                        tol=0,  # Length of cache array
                    )
                    sys = control.StateSpace(A, B, C, D, dt=0)
                    tf_l_array[row, col] = control.ss2tf(sys)
        # Fit SISO transfer functions of right scale
        for row in range(D_r_omega.shape[0]):
            for col in range(D_r_omega.shape[1]):
                if mask_r[row, col] == 0:
                    tf_r_array[row, col] = control.TransferFunction([0], [1], dt=0)
                elif mask_r[row, col] == 1:
                    if isinstance(order, np.ndarray) and (orders_r[row, col] != 0):
                        warnings.warn(
                            "Entries of `order` in last uncertainty block "
                            "should be 0 since those transfer functions are "
                            "known to be 1. Ignoring value of "
                            f"`order[{row}, {col}]`."
                        )
                    tf_r_array[row, col] = control.TransferFunction([1], [1], dt=0)
                else:
                    n, A, B, C, D = slycot.sb10yd(
                        discfl=0,  # Continuous-time
                        flag=1,  # Constrain stable, minimum phase
                        lendat=omega.shape[0],
                        rfrdat=np.real(D_r_omega[row, col, :]),
                        ifrdat=np.imag(D_r_omega[row, col, :]),
                        omega=omega,
                        n=orders_r[row, col],
                        tol=0,  # Length of cache array
                    )
                    sys = control.StateSpace(A, B, C, D, dt=0)
                    tf_r_array[row, col] = control.ss2tf(sys)

        tf_l = control.combine_tf(tf_l_array)
        tf_r = control.combine_tf(tf_r_array)
        ss_l = control.tf2ss(tf_l)
        ss_r = control.tf2ss(tf_r)
        ss_r_inv = _invert_biproper_ss(ss_r)
        return ss_l, ss_r_inv


def _generate_d_scale_mask(
    block_structure: List[uncertainty_structure.UncertaintyBlock],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a mask for the D-scale fit for the block structure.

    Entries known to be zero are set to 0. Entries known to be one are set to
    1. Entries to be fit numerically are set to -1.

    Parameters
    ----------
    block_structure : List[uncertainty_structure.UncertaintyBlock]
        Uncertainty block structure description.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Array of integers indicating zero, one, and unknown elements in the
        block structure.
    """
    num_blocks = len(block_structure)
    idx_last_full_block = -1
    for idx, block in enumerate(reversed(block_structure)):
        if isinstance(block, uncertainty_structure.ComplexFullBlock):
            idx_last_full_block = len(block_structure) - 1 - idx
            break
    mask_l_lst = []
    mask_r_lst = []
    for i in range(num_blocks):
        # Uncertainty block
        block = block_structure[i]
        if (i == idx_last_full_block) and (not block.is_diagonal):
            # Last scaling is always identity if it is a full perturbation
            mask_l_lst.append(np.eye(block.num_perf_outputs, dtype=int))
            mask_r_lst.append(np.eye(block.num_exog_inputs, dtype=int))
        elif (not block.is_complex) and (not block.is_diagonal):
            raise NotImplementedError("Real full perturbations are not supported.")
        elif (not block.is_complex) and (block.is_diagonal):
            raise NotImplementedError(
                "Real diagonal perturbations are not yet supported."
            )
        elif (block.is_complex) and (block.is_diagonal):
            mask_l_lst.append(-1 * np.tri(block.num_perf_outputs, dtype=int).T)
            mask_r_lst.append(-1 * np.tri(block.num_perf_outputs, dtype=int).T)
        elif (block.is_complex) and (not block.is_diagonal):
            mask_l_lst.append(-1 * np.eye(block.num_perf_outputs, dtype=int))
            mask_r_lst.append(-1 * np.eye(block.num_exog_inputs, dtype=int))
    mask_l = scipy.linalg.block_diag(*mask_l_lst)
    mask_r = scipy.linalg.block_diag(*mask_r_lst)
    return mask_l, mask_r


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
