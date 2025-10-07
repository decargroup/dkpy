"""Uncertainty characterization utilties."""

__all__ = [
    "compute_uncertainty_residual_response",
    "compute_optimal_uncertainty_weight_response",
    "fit_overbounding_uncertainty_weight",
    "plot_magnitude_response_nom_offnom",
    "plot_phase_response_nom_offnom",
    "plot_singular_value_response_nom_offnom",
    "plot_singular_value_response_uncertainty_residual",
    "plot_singular_value_response_uncertainty_residual_comparison",
    "plot_magnitude_response_uncertainty_weight",
]

import warnings

import control
import numpy as np
import cvxpy
from matplotlib import pyplot as plt
from typing import List, Optional, Union, Tuple, Dict, Callable, Set, Any
import scipy

from . import utilities


def compute_uncertainty_residual_response(
    complex_response_nom: np.ndarray,
    complex_response_offnom_list: np.ndarray,
    uncertainty_model: Union[str, List[str], Set[str]] = {
        "A",
        "I",
        "O",
        "iA",
        "iI",
        "iO",
    },
    tol_residual_existence: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Compute the residual response of unstructured uncertainty models.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Frequency response of the nominal system.
    complex_response_offnom_list : np.ndarray
        Frequency response of the off-nominal system.
    uncertainty_model : Union[str, List[str], Set[str]]
        Uncertainty model identifiers to compute the residual response.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    Union[control.FrequencyResponseList, Dict[str, control.FrequencyResponseList]]
        Frequency response of the uncertainty residuals.

    Raises
    ------
    ValueError
        An invalid uncertainty model identifier was found in `uncertainty_model`.

    Examples
    --------
    Compute the residuals for the six unstructured uncertainty models from the nominal
    and off-nominal frequency responses.

    >>> complex_response_nom, complex_response_offnom_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_model = {"A", "I", "O", "iA", "iI", "iO"}
    >>> complex_response_residual_dict = compute_uncertainty_residual_response(
    ...     complex_response_nom,
    ...     complex_response_offnom_list,
    ...     uncertainty_model
    ... )
    """

    # Uncertainty residual response dictionary
    frequency_response_residual_dict = {}

    # Check uncertainty model identifiers
    uncertainty_model = set(uncertainty_model)
    valid_uncertainty_model = {"A", "I", "O", "iA", "iI", "iO"}
    if not uncertainty_model.issubset(valid_uncertainty_model):
        raise ValueError(
            "The uncertainty model identifiers provided in `uncertainty_model` do not "
            "all correspond to valid uncertainty models. In particular, "
            f"{uncertainty_model.difference(valid_uncertainty_model)} are not valid "
            "uncertainty model identifiers. The identifiers are:\n"
            '\t"A": Additive uncertainty\n'
            '\t"I": Multiplicative input uncertainty\n'
            '\t"O": Multiplicative output uncertainty\n'
            '\t"iA": Inverse additive uncertainty\n'
            '\t"iI": Inverse multiplicative input uncertainty\n'
            '\t"iO": Inverse multiplicative output uncertainty'
        )

    # Additive uncertainty residual response
    if "A" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_additive_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["A"] = frequency_response_residual_list

    # Multiplicative input uncertainty residual response
    if "I" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_multiplicative_input_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["I"] = frequency_response_residual_list

    # Multiplicative output uncertainty residual response
    if "O" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_multiplicative_output_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["O"] = frequency_response_residual_list

    # Inverse additive uncertainty residual response
    if "iA" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_inverse_additive_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["iA"] = frequency_response_residual_list

    # Inverse multiplicative input uncertainty residual response
    if "iI" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_inverse_multiplicative_input_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["iI"] = frequency_response_residual_list

    # Inverse multiplicative output uncertainty residual response
    if "iO" in uncertainty_model:
        frequency_response_residual_list = _compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            _compute_uncertainty_residual_inverse_multiplicative_output_freq,
            tol_residual_existence,
        )
        frequency_response_residual_dict["iO"] = frequency_response_residual_list

    return frequency_response_residual_dict


def _compute_uncertainty_residual_response(
    complex_response_nom: np.ndarray,
    complex_response_offnom_list: np.ndarray,
    compute_uncertainty_residual_freq: Callable,
    tol_residual_existence: float,
) -> np.ndarray:
    """Compute the uncertainty residual response for a given uncertainty model.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Frequency response of the nominal system.
    complex_response_offnom_list : np.ndarray
        Frequency response of the off-nominal system.
    compute_uncertainty_residual_freq : Callable,
        Uncertainty residual computation function at a given frequency.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    control.FrequencyResponseList
        Uncertainty residual response of the given uncertainty model.
    """

    # Frequency response parameters
    num_offnom = complex_response_offnom_list.shape[0]
    num_frequency = complex_response_offnom_list.shape[1]

    complex_response_residual_list = []
    for idx_offnom in range(num_offnom):
        complex_response_residual = []
        complex_response_offnom = complex_response_offnom_list[idx_offnom, :, :, :]
        for idx_freq in range(num_frequency):
            complex_response_nom_freq = complex_response_nom[idx_freq, :, :]
            complex_response_offnom_freq = complex_response_offnom[idx_freq, :, :]
            complex_response_residual_freq = compute_uncertainty_residual_freq(
                complex_response_nom_freq,
                complex_response_offnom_freq,
                tol_residual_existence,
            )
            complex_response_residual.append(complex_response_residual_freq)
        complex_response_residual = np.array(complex_response_residual, dtype=complex)
        complex_response_residual_list.append(complex_response_residual)
    complex_response_residual_list = np.array(complex_response_residual_list)

    return complex_response_residual_list


def _compute_uncertainty_residual_additive_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: Optional[float] = None,
) -> np.ndarray:
    """Compute the additive uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Additive uncertainty residual at a given frequency.
    """
    complex_response_residual_freq = (
        complex_response_offnom_freq - complex_response_nom_freq
    )

    return complex_response_residual_freq


def _compute_uncertainty_residual_multiplicative_input_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-8,
):
    """Compute the multiplicative input uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative input uncertainty residual at a given frequency.

    Raises
    ------
    ValueError
        An input multiplicative uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    num_inputs = complex_response_nom_freq.shape[1]
    num_outputs = complex_response_nom_freq.shape[0]
    if num_inputs < num_outputs:
        warnings.warn(
            "Multipliative input uncertainty models cannot include all possible "
            "off-nominal systems when the number of inputs is less than the number of "
            "outputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )

    A = complex_response_nom_freq
    B = complex_response_offnom_freq - complex_response_nom_freq
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    complex_response_residual_freq = X

    if num_inputs >= num_outputs:
        return complex_response_residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_response_residual_freq
        else:
            raise ValueError(
                "A multiplicative input uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is due "
                "to the fact that the multiplicative input uncertainty model cannot "
                "account for all off-nominal models when the number of inputs is less "
                "than the number of outputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_uncertainty_residual_multiplicative_output_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-8,
):
    """Compute the multiplicative output uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative output uncertainty residual at a given frequency.

    Raises
    ------
    ValueError
        An output multiplicative uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    num_inputs = complex_response_nom_freq.shape[1]
    num_outputs = complex_response_nom_freq.shape[0]
    if num_inputs > num_outputs:
        warnings.warn(
            "Multipliative output uncertainty models cannot include all possible "
            "off-nominal systems when the number of outputs is less than the number of "
            "inputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )
    A = complex_response_nom_freq.T
    B = complex_response_offnom_freq.T - complex_response_nom_freq.T
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    complex_response_residual_freq = X.T

    if num_inputs <= num_outputs:
        return complex_response_residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_response_residual_freq
        else:
            raise ValueError(
                "A multiplicative output uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is "
                "due to the fact that the multiplicative output uncertainty model "
                "cannot account for all off-nominal models when the number of outputs "
                "is less than the number of inputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_uncertainty_residual_inverse_additive_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-8,
):
    """Compute the inverse additive uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a inverse additive uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse additive uncertainty residual at a given frequency.

    Raises
    ------
    ValueError
        An inverse additive uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    num_inputs = complex_response_nom_freq.shape[1]
    num_outputs = complex_response_nom_freq.shape[0]
    if num_inputs != num_outputs:
        warnings.warn(
            "Inverse additive uncertainty models cannot include all possible "
            "off-nominal systems when the number of inputs is not equal to the number "
            "of outputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )

    A1 = complex_response_offnom_freq
    B1 = complex_response_offnom_freq - complex_response_nom_freq
    Y, residues_lstsq_1, _, _ = scipy.linalg.lstsq(A1, B1)
    A2 = complex_response_nom_freq.T
    B2 = Y.T
    X, residues_lstsq_2, _, _ = scipy.linalg.lstsq(A2, B2)
    complex_response_residual_freq = X.T

    if num_inputs == num_outputs:
        return complex_response_residual_freq
    else:
        if np.all(residues_lstsq_1 <= tol_residual_existence) and np.all(
            residues_lstsq_2 <= tol_residual_existence
        ):
            return complex_response_residual_freq
        else:
            raise ValueError(
                "An inverse additive uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is "
                "due to the fact that the inverse additive uncertainty model "
                "cannot account for all off-nominal models when the number of inputs "
                "not equal to the number of outputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_uncertainty_residual_inverse_multiplicative_input_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-8,
):
    """Compute the inverse multiplicative input uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq: np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of an inverse multiplicative input uncertainty
        residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative input uncertainty residual at a given frequency.

    Raises
    ------
    ValueError
        An inverse multiplicative input uncertainty residual does not exist at the
        given frequency as the linear system used to compute the residual does not have
        a solution.
    """

    num_inputs = complex_response_nom_freq.shape[1]
    num_outputs = complex_response_nom_freq.shape[0]
    if num_inputs < num_outputs:
        warnings.warn(
            "Inverse multipliative input uncertainty models cannot include all "
            "possible off-nominal systems when the number of inputs is less than the "
            "number of outputs. This may result in an error if an uncertainty residual "
            "cannot be found for a given nominal and off-nominal frequency response "
            "matrix."
        )

    A = complex_response_offnom_freq
    B = complex_response_offnom_freq - complex_response_nom_freq
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    complex_response_residual_freq = X

    if num_inputs >= num_outputs:
        return complex_response_residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_response_residual_freq
        else:
            raise ValueError(
                "An inverse multiplicative input uncertainty residual does not exist "
                "for the given nominal and off-nominal frequency response matrix. This "
                "is due to the fact that the inverse multiplicative input uncertainty "
                "model cannot account for all off-nominal models when the number of "
                "inputs is less than the number of outputs. Please consider using a "
                "different uncertainty model."
            )


def _compute_uncertainty_residual_inverse_multiplicative_output_freq(
    complex_response_nom_freq: np.ndarray,
    complex_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-8,
):
    """Compute the inverse multiplicative output uncertainty residual at a frequency.

    Parameters
    ----------
    complex_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    complex_response_offnom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative output uncertainty residual at a given frequency.

    Raises
    ------
    ValueError
        An inverse multiplicative output uncertainty residual does not exist at the
        given frequency as the linear system used to compute the residual does not have
        a solution.
    """

    num_inputs = complex_response_nom_freq.shape[1]
    num_outputs = complex_response_nom_freq.shape[0]
    if num_inputs > num_outputs:
        warnings.warn(
            "Inverse multipliative output uncertainty models cannot include all "
            "possible off-nominal systems when the number of outputs is less than the "
            "number of inputs. This may result in an error if an uncertainty residual "
            "cannot be found for a given nominal and off-nominal frequency response "
            "matrix."
        )
    A = complex_response_offnom_freq.T
    B = complex_response_offnom_freq.T - complex_response_nom_freq.T
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    complex_response_residual_freq = X.T

    if num_inputs <= num_outputs:
        return complex_response_residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_response_residual_freq
        else:
            raise ValueError(
                "An inverse multiplicative output uncertainty residual does not exist "
                "for the given nominal and off-nominal frequency response matrix. This "
                "is due to the fact that the inverse multiplicative output uncertainty "
                "model cannot account for all off-nominal models when the number of "
                "outputs is less than the number of inputs. Please consider using a "
                "different uncertainty model."
            )


# NOTE: At the moment, the function returns a 2D array representing the frequency
# response of the weight. However given that the weights are always diagonal, I'm not
# sure if it would be better to return a 1D array of just the diagonal elements.
def compute_optimal_uncertainty_weight_response(
    complex_response_residual_list: np.ndarray,
    weight_left_structure: str,
    weight_right_structure: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal uncertainty weight frequency response.

    Parameters
    ----------
    complex_response_residual_list : np.ndarray
        Frequency response of the residuals for which to compute the optimal uncertainty
        weights.
    weight_left_structure : str
        Structure of the left uncertainty weight. Valid structures include: "diagonal",
        "scalar", and "identity".
    weight_right_structure : str
        Structure of the right uncertainty weight. Valid structures include: "diagonal",
        "scalar", and "identity".

    Returns
    -------
    Tuple[control.FrequencyResponseData, control.FrequencyResponse]
        Frequency response of the diagonal elements of the left and right uncertainty
        weights.

    Examples
    --------
    Compute the optimal left and right uncertainty weights for the multiplicative input
    residual frequency response where the left and right uncertainty weights are assumed
    to be diagonal.

    >>> complex_response_nom, complex_response_offnom_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_model = {"A", "I", "O", "iA", "iI", "iO"}
    >>> complex_response_residual_dict = compute_uncertainty_residual_response(
    ...     complex_response_nom,
    ...     complex_response_offnom_list,
    ...     uncertainty_model
    ... )
    >>> complex_response_weight_left, complex_response_weight_right = (
    ...     dkpy.compute_optimal_uncertainty_weight_response(
    ...         complex_response_residual_dict["I"], "diagonal", "diagonal"
    ...     )
    ... )
    """

    # Frequency response parameters
    num_frequency = complex_response_residual_list.shape[1]

    # Compute optimal uncertainty weights
    complex_response_weight_left = []
    complex_response_weight_right = []
    for idx_freq in range(num_frequency):
        complex_residual_freq = complex_response_residual_list[:, idx_freq, :, :]
        weight_left_freq, weight_right_freq = _compute_optimal_weight_freq(
            complex_residual_freq, weight_left_structure, weight_right_structure
        )
        complex_response_weight_left.append(weight_left_freq)
        complex_response_weight_right.append(weight_right_freq)

    # Generate uncertainty weight complex frequency reponses
    complex_response_weight_left = np.array(complex_response_weight_left)
    complex_response_weight_right = np.array(complex_response_weight_right)

    return complex_response_weight_left, complex_response_weight_right


def _compute_optimal_weight_freq(
    complex_residual_offnom_set_freq: np.ndarray,
    weight_left_structure: str,
    weight_right_structure: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal uncertainty weight at a given frequency.

    Parameters
    ----------
    complex_residual_offnom_set_freq : np.ndarray
        Frequency response matrix of the off-nominal models at a given frequency.
    weight_left_structure : str
        Structure of the left uncertainty weight. Valid structures include: "diagonal",
        "scalar", and "identity".
    weight_right_structure : str
        Structure of the right uncertainty weight. Valid structures include: "diagonal",
        "scalar", and "identity".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The left and right uncertainty weight frequency response matrices at a given
        frequency.
    """

    # System parameters
    num_left = complex_residual_offnom_set_freq.shape[1]
    num_right = complex_residual_offnom_set_freq.shape[2]
    num_offnom = complex_residual_offnom_set_freq.shape[0]

    # HACK: Find a better way to parse the structure assumptions of the weights instead
    # of using strings

    # Generate left weight variable
    if weight_left_structure == "diagonal":
        L = cvxpy.Variable((num_left, num_left), diag=True)
    elif weight_left_structure == "scalar":
        L_scalar = cvxpy.Variable()
        L = L_scalar * scipy.sparse.eye_array(num_left)
    elif weight_left_structure == "identity":
        L = cvxpy.Parameter(
            shape=(num_left, num_left), value=np.eye(num_left), diag=True
        )
    else:
        raise ValueError(
            f'"{weight_left_structure}" is not a valid value for '
            '`weight_right_structure`. It must take a value of either "diagonal" '
            '"scalar", or "identity".'
        )

    # Generate right weight variable
    if weight_right_structure == "diagonal":
        R = cvxpy.Variable((num_right, num_right), diag=True)
    elif weight_right_structure == "scalar":
        R_scalar = cvxpy.Variable()
        R = R_scalar * scipy.sparse.eye_array(num_right)
    elif weight_right_structure == "identity":
        R = cvxpy.Parameter(
            shape=(num_right, num_right), value=np.eye(num_right), diag=True
        )
    else:
        raise ValueError(
            f'"{weight_right_structure}" is not a valid value for '
            '`weight_right_structure`. It must take a value of either "diagonal" '
            '"scalar", or "identity".'
        )

    # Generate optimal uncertainty weight constraints
    constraint_freq_list = []
    for idx_offnom in range(num_offnom):
        E_k = cvxpy.Parameter(
            shape=(num_left, num_right),
            value=complex_residual_offnom_set_freq[idx_offnom, :, :],
            complex=True,
        )
        constraint_matrix_freq = cvxpy.bmat(
            [
                [L, E_k],
                [E_k.H, R],
            ]
        )
        constraint_freq = constraint_matrix_freq >> 0
        constraint_freq_list.append(constraint_freq)
    constraint_freq_list.append(L.H == L)
    constraint_freq_list.append(L >> 0)
    constraint_freq_list.append(R.H == R)
    constraint_freq_list.append(R >> 0)

    # Semidefinite program
    objective = cvxpy.Minimize(cvxpy.trace(L) + cvxpy.trace(R))
    problem = cvxpy.Problem(objective, constraint_freq_list)
    problem.solve(solver="MOSEK")

    # Extract left weight
    if weight_left_structure == "identity":
        L_value = np.array(L.value)
    else:
        L_value = np.array(L.value.toarray())
    complex_response_weight_left_freq = np.sqrt(L_value)
    # Extract right weight
    if weight_right_structure == "identity":
        R_value = np.array(R.value)
    else:
        R_value = np.array(R.value.toarray())
    complex_response_weight_right_freq = np.sqrt(R_value)

    return complex_response_weight_left_freq, complex_response_weight_right_freq


# NOTE: I'm not sure if the representation of the `weight` variable is intuitive as it
# is a 1D array for the diagonal elements of the uncertainty weight response, which
# uses a 2D array for its description.
def fit_overbounding_uncertainty_weight(
    complex_response_uncertainty_weight: np.ndarray,
    omega: np.ndarray,
    order: Union[int, List[int], np.ndarray],
    weight: Optional[np.ndarray] = None,
    linear_solver_param: Dict[str, Any] = {},
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    num_spec_constr: int = 500,
) -> control.StateSpace:
    """
    Fit an overbounding stable and minimum-phase state-space uncertainty weight to
    frequency response data.

    Parameters
    ----------
    complex_response_uncertainty_weight : np.ndarray
        Uncertainty weight frequency response used for the overbounding fit.
    order : Union[int, List[int], np.ndarray]
        Order of the LTI system fit. If `order` is an `int`, the order will be
        used for all elements of the weight. If `order` is a `List` or `np.ndarray`,
        the order can be specified for each element of the weight.
    weight : Optional[np.ndarray] = None
        Frequency-dependent weight used to improve the fit over certain bandwidths. The
        weight is a 2D array with the first dimension representing the number of
        elements in the weight and the second dimension representing the number of
        frequency points.
    linear_solver_param : Dict[str, Any]
        Keyword arguments for the linear feasibility problem solver. See
        [#cvxpy_solver]_ for more information.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum allowable number of iterations in the bisection algorithm.
    num_spec_constr : int
        Number of constraints used to enforce the spectral factorizability of the
        fitted autocorrelation.

    Returns
    -------
    control.StateSpace
        Fitted overbounding uncertainty weight state-space systems.

    Examples
    --------
    Fit an overbounding stable and minimum phase system to the left and right
    uncertainty weights for the multiplicative input uncertainty model.

    >>> complex_response_nom, complex_response_offnom_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_model = {"A", "I", "O", "iA", "iI", "iO"}
    >>> complex_response_residual_dict = compute_uncertainty_residual_response(
    ...     complex_response_nom,
    ...     complex_response_offnom_list,
    ...     uncertainty_model
    ... )
    >>> complex_response_weight_left, complex_response_weight_right = (
    ...     dkpy.compute_optimal_uncertainty_weight_response(
    ...         complex_response_residual_dict["I"], "diagonal", "diagonal"
    ...     )
    ... )
    >>> weight_left = dkpy.fit_overbounding_uncertainty_weight(
    ...     complex_response_weight_left, omega, [4, 5]
    ... )
    >>> weight_right = dkpy.fit_overbounding_uncertainty_weight(
    ...     complex_response_weight_right, omega, [3, 5]
    ... )

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Parse arguments
    num_elements = complex_response_uncertainty_weight.shape[1]
    order_list = (
        order * np.ones(num_elements, dtype=int)
        if isinstance(order, int)
        else np.array(order, dtype=int)
    )

    # NOTE: Is this a sensible default?
    if weight is None:
        # Take the default frequency-dependent weight as the normalized magnitude the
        # uncertainty weight magnitude in order to place greater importance on tightly
        # overbounding at the largest uncertainties
        weight = np.diagonal(
            np.abs(complex_response_uncertainty_weight), axis1=1, axis2=2
        )
        weight = weight / np.max(weight, axis=0)

    uncertainty_weight_list = []
    for idx_element in range(num_elements):
        # Extract the parameters relevant to each SISO uncertainty weight element
        magnitude_response_weight_element = np.abs(
            complex_response_uncertainty_weight[:, idx_element, idx_element]
        )
        order_element = order_list[idx_element]
        weight_element = weight[:, idx_element]

        # Fit the uncertainty weight to each SISO element
        uncertainty_weight_element = utilities._fit_magnitude_log_chebyshev_siso(
            omega=omega,
            magnitude_fit=magnitude_response_weight_element,
            order=order_element,
            magnitude_lower_bound=magnitude_response_weight_element,
            weight=weight_element,
            linear_solver_param=linear_solver_param,
            tol_bisection=tol_bisection,
            max_iter_bisection=max_iter_bisection,
            num_spec_constr=num_spec_constr,
        )
        uncertainty_weight_list.append(uncertainty_weight_element)

    # Construct uncertainty weight fit from SISO elements
    uncertainty_weight = control.append(*uncertainty_weight_list)

    return uncertainty_weight


# TODO: Increase customizability of plot
# For example, include the ability for absolute or dB
def plot_magnitude_response_nom_offnom(
    complex_response_nom: np.ndarray,
    complex_response_offnom_list: np.ndarray,
    omega: np.ndarray,
):
    """Plot magnitude response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : np.ndarray
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    """

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]
    num_outputs = complex_response_offnom_list.shape[2]
    num_inputs = complex_response_offnom_list.shape[3]

    # Initialize figure
    fig, ax = plt.subplots(
        num_outputs, num_inputs, sharex=True, layout="constrained", squeeze=False
    )

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        magnitude_offnom = np.abs(complex_response_offnom_list[idx_offnom, :, :, :])
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].semilogx(
                    omega,
                    control.mag2db(magnitude_offnom[:, idx_output, idx_input]),
                    color="tab:orange",
                    alpha=0.25,
                    label="Off-nominal",
                )

    # Nominal frequency response
    magnitude_nom = np.abs(complex_response_nom)
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].semilogx(
                omega,
                control.mag2db(magnitude_nom[:, idx_output, idx_input]),
                color="tab:blue",
                alpha=1.0,
                label="Nominal",
            )

    # Plot settings
    for ax_output in ax:
        for ax_output_input in ax_output:
            ax_output_input.set_ylabel("Magnitude (dB)")
            ax_output_input.grid()
    for idx_input in range(num_inputs):
        ax[-1, idx_input].set_xlabel("$\\omega$ (rad/s)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )


# TODO: Increase customizability of plot
# For example, include the ability for deg or rad
def plot_phase_response_nom_offnom(
    complex_response_nom: np.ndarray,
    complex_response_offnom_list: np.ndarray,
    omega: np.ndarray,
):
    """Plot phase response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : np.ndarray
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    """

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]
    num_outputs = complex_response_offnom_list.shape[2]
    num_inputs = complex_response_offnom_list.shape[3]

    # Initialize figure
    fig, ax = plt.subplots(
        num_outputs, num_inputs, sharex=True, layout="constrained", squeeze=False
    )

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        phase_offnom = np.angle(complex_response_offnom_list[idx_offnom, :, :, :])
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].semilogx(
                    omega,
                    180 / np.pi * phase_offnom[:, idx_output, idx_input],
                    color="tab:orange",
                    alpha=0.25,
                    label="Off-nominal",
                )

    # Nominal frequency response
    phase_nom = np.angle(complex_response_nom)
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].semilogx(
                omega,
                180 / np.pi * phase_nom[:, idx_output, idx_input],
                color="tab:blue",
                alpha=1.0,
                label="Nominal",
            )

    # Plot settings
    for ax_output in ax:
        for ax_output_input in ax_output:
            ax_output_input.set_ylabel("Phase (deg)")
            ax_output_input.grid()
    for idx_input in range(num_inputs):
        ax[-1, idx_input].set_xlabel("$\\omega$ (rad/s)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )


# TODO: Increase customizability of plot
def plot_singular_value_response_nom_offnom(
    complex_response_nom: np.ndarray,
    complex_response_offnom_list: np.ndarray,
    omega: np.ndarray,
):
    """Plot singular value response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : np.ndarray
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    """

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]

    # Initialize figure
    fig, ax = plt.subplots(layout="constrained")

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        sval_offnom = np.linalg.svdvals(
            complex_response_offnom_list[idx_offnom, :, :, :]
        )
        for idx_sval in range(sval_offnom.shape[1]):
            ax.semilogx(
                omega,
                control.mag2db(sval_offnom[:, idx_sval]),
                color="tab:orange",
                alpha=0.5,
                label="Off-nominal",
            )

    # Nominal frequency response
    sval_nom = np.linalg.svdvals(complex_response_nom)
    for idx_sval in range(sval_nom.shape[1]):
        ax.semilogx(
            omega,
            control.mag2db(sval_nom[:, idx_sval]),
            color="tab:blue",
            alpha=1.0,
            label="Nominal",
        )

    # Plot settings
    ax.set_ylabel("Magnitude (dB)")
    ax.grid()
    ax.set_xlabel("$\\omega$ (rad/s)")
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )


# TODO: Increase customizability of plot
def plot_singular_value_response_uncertainty_residual(
    complex_response_residual_dict: Dict[str, np.ndarray],
    omega: np.ndarray,
):
    """Plot the singular value response of the uncertainty residuals for different
    unstructured uncertainty models on separate figures.

    Parameters
    ----------
    complex_response_residual_dict : Dict[str, np.ndarray]
        Dictionary of the uncertainty residual frequency response matrices over a grid
        of frequencies for different uncertainty models.
    omega : np.narray
        Angular frequency grid.
    """

    # Iterate over each uncertainty model
    for (
        uncertainty_model_id,
        complex_response_residual,
    ) in complex_response_residual_dict.items():
        # Off-nominal frequency response parameters
        num_offnom = complex_response_residual.shape[0]

        # Compute the singular value and maximum singlar value response of the residuals
        sval_response_residual = np.linalg.svdvals(complex_response_residual)
        sval_max_response_residual = np.max(sval_response_residual, axis=(0, 2))

        # Intialize the plot
        fig, ax = plt.subplots()

        # Singular value response of the residuals
        for idx_offnom in range(num_offnom):
            for idx_sval in range(sval_response_residual.shape[2]):
                ax.semilogx(
                    omega,
                    control.mag2db(sval_response_residual[idx_offnom, :, idx_sval]),
                    color="grey",
                    alpha=0.5,
                    label=f"$\\sigma(E_{{{uncertainty_model_id}}})$",
                )
        # Maximum singular value response of the residuals
        ax.semilogx(
            omega,
            control.mag2db(sval_max_response_residual),
            color="black",
            label=f"$\\max \\; \\sigma(E_{{{uncertainty_model_id}}})$",
        )

        # Plot settings
        ax.set_ylabel("Magnitude (dB)")
        ax.grid()
        ax.set_xlabel("$\\omega$ (rad/s)")
        handles, labels = ax.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))
        fig.legend(
            labels=legend_dict.keys(),
            handles=legend_dict.values(),
            loc="outside lower center",
            ncol=2,
        )


# TODO: Increase customizability of plot
def plot_singular_value_response_uncertainty_residual_comparison(
    complex_response_residual_dict: Dict[str, np.ndarray],
    omega: np.ndarray,
):
    """Plot the maximum singular value response of the uncertainty residuals for
    different unstructured uncertainty models on the same figure for comparision of the
    various uncertainty models.

    Parameters
    ----------
    complex_response_residual_dict : Dict[str, np.ndarray]
        Dictionary of the uncertainty residual frequency response matrices over a grid
        of frequencies for different uncertainty models.
    omega : np.narray
        Angular frequency grid.
    """

    # Maximum singular value response of uncertainty residuals
    sval_max_response_residual_dict = {}
    for (
        uncertainty_model_id,
        complex_response_residual_list,
    ) in complex_response_residual_dict.items():
        sval_response_residual_list = np.linalg.svdvals(complex_response_residual_list)
        sval_max_response_residual = np.max(sval_response_residual_list, axis=(0, 2))
        sval_max_response_residual_dict[uncertainty_model_id] = (
            sval_max_response_residual
        )

    # Initialize figure
    fig, ax = plt.subplots()

    # Maximum singular value reponse of uncertainty residuals
    for (
        uncertainty_model_id,
        sval_max_response_residual,
    ) in sval_max_response_residual_dict.items():
        ax.semilogx(
            omega,
            control.mag2db(sval_max_response_residual),
            label=f"$\\max \\; {{\\sigma}}(E_{{{uncertainty_model_id}}})$",
        )

    # Plot settings
    ax.set_ylabel("Magnitude (dB)")
    ax.grid()
    ax.set_xlabel("$\\omega$ (rad/s)")
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=6,
    )


def plot_magnitude_response_uncertainty_weight(
    complex_response_weight_left: np.ndarray,
    complex_response_weight_right: np.ndarray,
    omega: np.ndarray,
    weight_left: Optional[control.StateSpace] = None,
    weight_right: Optional[control.StateSpace] = None,
):
    """Plot the diagonal elements of the optimal left and right uncertainty weight
    frequency responses. Optionally, the fitted overbounding left and right uncertainty
    weights can also be displayed.

    Parameters
    ----------
    complex_response_weight_left : np.ndarray,
        Frequency response matrices of the left uncertainty weight over a grid of
        frequencies.
    complex_response_weight_right : np.ndarray,
        Frequency response matrices of the right uncertainty weight over a grid of
        frequencies.
    omega : np.ndarray
        Angular frequency grid.
    weight_left: Optional[control.StateSpace] = None,
        State-space model if the fitted overbounding left uncertainty weight.
    weight_right: Optional[control.StateSpace] = None,
        State-space model if the fitted overbounding right uncertainty weight.
    """

    # Uncertainty weight parameters
    num_left = complex_response_weight_left.shape[1]
    num_right = complex_response_weight_right.shape[1]

    # Magnitude response of the uncertainty weights
    magnitude_response_weight_left = np.abs(complex_response_weight_left)
    magnitude_response_weight_right = np.abs(complex_response_weight_right)

    # Initialize figure
    fig, ax = plt.subplots(
        max(num_left, num_right), 2, sharex=True, layout="constrained"
    )

    # Plot left uncertainty weight frequency response
    for idx_left in range(num_left):
        ax[idx_left, 0].semilogx(
            omega,
            control.mag2db(magnitude_response_weight_left[:, idx_left, idx_left]),
            linestyle="",
            marker="*",
            color="tab:blue",
            label="Response",
        )
        ax[idx_left, 0].set_ylabel("$|W_{L, (1, 1)}|$ (dB)")
        ax[idx_left, 0].grid()
    # Plot right uncertainty weight frequency response
    for idx_right in range(num_left):
        ax[idx_right, 1].semilogx(
            omega,
            control.mag2db(magnitude_response_weight_right[:, idx_right, idx_right]),
            linestyle="",
            marker="*",
            color="tab:blue",
            label="Response",
        )
        ax[idx_right, 1].set_ylabel("$|W_{R, (1, 1)}|$ (dB)")
        ax[idx_right, 1].grid()

    # Plot left uncertainty weight fit frequency response
    if weight_left is not None:
        response_fit_weight_left = control.frequency_response(weight_left, omega)
        magnitude_response_fit_weight_left = np.array(
            response_fit_weight_left.magnitude
        )
        for idx_left in range(num_left):
            ax[idx_left, 0].semilogx(
                omega,
                control.mag2db(
                    magnitude_response_fit_weight_left[idx_left, idx_left, :]
                ),
                color="tab:orange",
                label="Fit",
            )
    # Plot right uncertainty weight fit frequency response
    if weight_right is not None:
        response_fit_weight_right = control.frequency_response(weight_right, omega)
        magnitude_response_fit_weight_right = np.array(
            response_fit_weight_right.magnitude
        )
        for idx_right in range(num_right):
            ax[idx_right, 1].semilogx(
                omega,
                control.mag2db(
                    magnitude_response_fit_weight_right[idx_right, idx_right, :]
                ),
                color="tab:orange",
                label="Fit",
            )

    # Plot settings
    for idx_col in range(2):
        ax[-1, idx_col].set_xlabel("$\\omega$ (rad/s)")
    for ax_row in ax:
        for ax_row_col in ax_row:
            if not ax_row_col.has_data():
                fig.delaxes(ax_row_col)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )
