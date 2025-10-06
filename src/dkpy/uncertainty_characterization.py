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
    frequency_response_nom: control.FrequencyResponseData,
    frequency_response_offnom_list: control.FrequencyResponseList,
    uncertainty_model: Union[str, List[str], Set[str]],
    tol_residual_existence: float = 1e-12,
) -> Dict[str, control.FrequencyResponseList]:
    """Compute the residual response of unstructured uncertainty models.

    Parameters
    ----------
    frequency_response_nom : control.FrequencyResponseData
        Frequency response of the nominal system.
    frequency_response_offnom_list : control.FrequencyResponseList
        Frequency response of the off-nominal system.
    uncertainty_model : Union[str, List[str], Set[str]]
        Uncertainty model identifiers to compute the residual response.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    Union[control.FrequencyResponseList, Dict[str, control.FrequencyResponseList]]
        Frequency response of the uncertainty residuals.
    """

    # Uncertainty residual response dictionary
    uncertainty_residual_response_dict = {}

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
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_additive_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["A"] = residual_response_list

    # Multiplicative input uncertainty residual response
    if "I" in uncertainty_model:
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_multiplicative_input_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["I"] = residual_response_list

    # Multiplicative output uncertainty residual response
    if "O" in uncertainty_model:
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_multiplicative_output_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["O"] = residual_response_list

    # Inverse additive uncertainty residual response
    if "iA" in uncertainty_model:
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_inverse_additive_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["iA"] = residual_response_list

    # Inverse multiplicative input uncertainty residual response
    if "iI" in uncertainty_model:
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_inverse_multiplicative_input_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["iI"] = residual_response_list

    # Inverse multiplicative output uncertainty residual response
    if "iO" in uncertainty_model:
        residual_response_list = _compute_uncertainty_residual_response(
            frequency_response_nom,
            frequency_response_offnom_list,
            _compute_uncertainty_residual_inverse_multiplicative_output_freq,
            tol_residual_existence,
        )
        uncertainty_residual_response_dict["iO"] = residual_response_list

    return uncertainty_residual_response_dict


def _compute_uncertainty_residual_response(
    frequency_response_nom: control.FrequencyResponseData,
    frequency_response_offnom_list: control.FrequencyResponseList,
    compute_uncertainty_residual_freq: Callable,
    tol_residual_existence: float,
) -> control.FrequencyResponseList:
    """Compute the uncertainty residual response for a given uncertainty model.

    Parameters
    ----------
    frequency_response_nom : control.FrequencyResponseData
        Frequency response of the nominal system.
    frequency_response_offnom_list : control.FrequencyResponseList
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
    frequency = frequency_response_nom.frequency
    residual_response_list = control.FrequencyResponseList()
    for frequency_response_offnom in frequency_response_offnom_list:
        residual_response = []
        response_complex_nom = frequency_response_nom.complex
        response_complex_offnom = frequency_response_offnom.complex
        for idx_freq in range(frequency.size):
            response_complex_nom_freq = response_complex_nom[:, :, idx_freq]
            response_complex_offnom_freq = response_complex_offnom[:, :, idx_freq]
            residual_response_freq = compute_uncertainty_residual_freq(
                response_complex_nom_freq,
                response_complex_offnom_freq,
                tol_residual_existence,
            )
            residual_response.append(residual_response_freq)
        residual_response = np.array(residual_response, dtype=complex).transpose(
            1, 2, 0
        )
        residual_response = control.FrequencyResponseData(residual_response, frequency)
        residual_response_list.append(residual_response)

    return residual_response_list


def _compute_uncertainty_residual_additive_freq(
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: Optional[float] = None,
) -> np.ndarray:
    """Compute the additive uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Additive uncertainty residual at a given frequency.
    """
    residual_freq = frequency_response_offnom_freq - frequency_response_nom_freq

    return residual_freq


def _compute_uncertainty_residual_multiplicative_input_freq(
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-12,
):
    """Compute the multiplicative input uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative input uncertainty residual at a given frequency.
    """

    num_inputs = frequency_response_nom_freq.shape[1]
    num_outputs = frequency_response_nom_freq.shape[0]
    if num_inputs < num_outputs:
        warnings.warn(
            "Multipliative input uncertainty models cannot include all possible "
            "off-nominal systems when the number of inputs is less than the number of "
            "outputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )

    A = frequency_response_nom_freq
    B = frequency_response_offnom_freq - frequency_response_nom_freq
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    residual_freq = X

    if num_inputs >= num_outputs:
        return residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return residual_freq
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
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-12,
):
    """Compute the multiplicative output uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative output uncertainty residual at a given frequency.
    """

    num_inputs = frequency_response_nom_freq.shape[1]
    num_outputs = frequency_response_nom_freq.shape[0]
    if num_inputs > num_outputs:
        warnings.warn(
            "Multipliative output uncertainty models cannot include all possible "
            "off-nominal systems when the number of outputs is less than the number of "
            "inputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )
    A = frequency_response_nom_freq.T
    B = frequency_response_offnom_freq.T - frequency_response_nom_freq.T
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    residual_freq = X.T

    if num_inputs <= num_outputs:
        return residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return residual_freq
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
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-12,
):
    """Compute the inverse additive uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a inverse additive uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse additive uncertainty residual at a given frequency.
    """

    num_inputs = frequency_response_nom_freq.shape[1]
    num_outputs = frequency_response_nom_freq.shape[0]
    if num_inputs != num_outputs:
        warnings.warn(
            "Inverse additive uncertainty models cannot include all possible "
            "off-nominal systems when the number of inputs is not equal to the number "
            "of outputs. This may result in an error if an uncertainty residual cannot "
            "be found for a given nominal and off-nominal frequency response matrix."
        )

    A1 = frequency_response_offnom_freq
    B1 = frequency_response_offnom_freq - frequency_response_nom_freq
    Y, residues_lstsq_1, _, _ = scipy.linalg.lstsq(A1, B1)
    A2 = frequency_response_nom_freq.T
    B2 = Y.T
    X, residues_lstsq_2, _, _ = scipy.linalg.lstsq(A2, B2)
    residual_freq = X.T

    if num_inputs == num_outputs:
        return residual_freq
    else:
        if np.all(residues_lstsq_1 <= tol_residual_existence) and np.all(
            residues_lstsq_2 <= tol_residual_existence
        ):
            return residual_freq
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
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-12,
):
    """Compute the inverse multiplicative input uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of an inverse multiplicative input uncertainty
        residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative input uncertainty residual at a given frequency.
    """

    num_inputs = frequency_response_nom_freq.shape[1]
    num_outputs = frequency_response_nom_freq.shape[0]
    if num_inputs < num_outputs:
        warnings.warn(
            "Inverse multipliative input uncertainty models cannot include all "
            "possible off-nominal systems when the number of inputs is less than the "
            "number of outputs. This may result in an error if an uncertainty residual "
            "cannot be found for a given nominal and off-nominal frequency response "
            "matrix."
        )

    A = frequency_response_offnom_freq
    B = frequency_response_offnom_freq - frequency_response_nom_freq
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    residual_freq = X

    if num_inputs >= num_outputs:
        return residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return residual_freq
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
    frequency_response_nom_freq: np.ndarray,
    frequency_response_offnom_freq: np.ndarray,
    tol_residual_existence: float = 1e-12,
):
    """Compute the inverse multiplicative output uncertainty residual at a frequency.

    Parameters
    ----------
    frequency_response_nom_freq : np.ndarray
        Nominal frequency response matrix evaluated at a given frequency.
    frequency_response_nom_freq : np.ndarray
        Frequency response matrix of a single off-nominal system evaluated at a given
        frequency.
    tol_residual_existence : float
        Tolerance for the existence of a multiplicative input uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative output uncertainty residual at a given frequency.
    """

    num_inputs = frequency_response_nom_freq.shape[1]
    num_outputs = frequency_response_nom_freq.shape[0]
    if num_inputs > num_outputs:
        warnings.warn(
            "Inverse multipliative output uncertainty models cannot include all "
            "possible off-nominal systems when the number of outputs is less than the "
            "number of inputs. This may result in an error if an uncertainty residual "
            "cannot be found for a given nominal and off-nominal frequency response "
            "matrix."
        )
    A = frequency_response_offnom_freq.T
    B = frequency_response_offnom_freq.T - frequency_response_nom_freq.T
    X, residues_lstsq, _, _ = scipy.linalg.lstsq(A, B)
    residual_freq = X.T

    if num_inputs <= num_outputs:
        return residual_freq
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return residual_freq
        else:
            raise ValueError(
                "An inverse multiplicative output uncertainty residual does not exist "
                "for the given nominal and off-nominal frequency response matrix. This "
                "is due to the fact that the inverse multiplicative output uncertainty "
                "model cannot account for all off-nominal models when the number of "
                "outputs is less than the number of inputs. Please consider using a "
                "different uncertainty model."
            )


def compute_optimal_uncertainty_weight_response(
    response_residual_list: control.FrequencyResponseList,
    weight_left_structure: str,
    weight_right_structure: str,
) -> Tuple[control.FrequencyResponseData, control.FrequencyResponseData]:
    """Compute the optimal uncertainty weight frequency response.

    Parameters
    ----------
    response_residual_list : control.FrequencyResponseList
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
    """
    # Frequency grid
    frequency = response_residual_list[0].frequency

    # Parse residual response data
    complex_response_residual_list = []
    for residual_response in response_residual_list:
        complex_response_residual = residual_response.complex
        complex_response_residual_list.append(complex_response_residual)
    complex_response_residual_array = np.array(complex_response_residual_list)

    # Compute optimal uncertainty weights
    complex_response_weight_left_list = []
    complex_response_weight_right_list = []
    for idx_freq in range(frequency.size):
        complex_residual_freq = complex_response_residual_array[:, :, :, idx_freq]
        weight_left_freq, weight_right_freq = _compute_optimal_weight_freq(
            complex_residual_freq, weight_left_structure, weight_right_structure
        )
        complex_response_weight_left_list.append(weight_left_freq)
        complex_response_weight_right_list.append(weight_right_freq)

    # Generate left uncertainty weight frequency response
    complex_response_weight_left_array = np.array(
        complex_response_weight_left_list
    ).transpose(1, 2, 0)
    response_weight_left = control.FrequencyResponseData(
        complex_response_weight_left_array, frequency
    )

    # Generate right uncertainty weight frequency response
    complex_response_weight_right_array = np.array(
        complex_response_weight_right_list
    ).transpose(1, 2, 0)
    response_weight_right = control.FrequencyResponseData(
        complex_response_weight_right_array, frequency
    )

    return response_weight_left, response_weight_right


def _compute_optimal_weight_freq(
    residual_offnom_set_freq: np.ndarray,
    weight_left_structure: str,
    weight_right_structure: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal uncertainty weight at a given frequency.

    Parameters
    ----------
    residual_offnom_set_freq : control.FrequencyResponseList
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
    num_left = residual_offnom_set_freq.shape[1]
    num_right = residual_offnom_set_freq.shape[2]
    num_offnom = residual_offnom_set_freq.shape[0]

    # if num_left == 1 and weight_left_structure == "diagonal":
    #     weight_left_structure = "scalar"
    # if num_right == 1 and weight_right_structure == "diagonal":
    #     weight_right_structure = "scalar"

    # HACK: Find a better way to parse the structure assumptions of the weights

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
            value=residual_offnom_set_freq[idx_offnom, :, :],
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
    weight_l = np.sqrt(L_value)
    # Extract right weight
    if weight_right_structure == "identity":
        R_value = np.array(R.value)
    else:
        R_value = np.array(R.value.toarray())
    weight_r = np.sqrt(R_value)

    return weight_l, weight_r


def fit_overbounding_uncertainty_weight(
    response_uncertainty_weight: control.FrequencyResponseData,
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
    response_uncertainty_weight : control.FrequencyResponseData
        Uncertainty weight frequency response used for the overbounding fit.
    order : Union[int, List[int], np.ndarray]
        Order of the LTI system fit. If `order` is an `int`, the order will be
        used for all elements of the weight. If `order` is a `List` or `np.ndarray`,
        the order can be specified for each element of the weight.
    weight : Optional[np.ndarray] = None
        Frequency-dependent weight used to improve the fit over certain bandwidths.
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

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Parse arguments
    num_elements = response_uncertainty_weight.ninputs
    omega = response_uncertainty_weight.frequency
    order_list = (
        order * np.ones(num_elements) if isinstance(order, int) else np.array(order)
    )

    # NOTE: Is this a sensible default?
    if weight is None:
        # Take the default frequency-dependent weight as the normalized magnitude the
        # uncertainty weight magnitude in order to place greater importance on tightly
        # overbounding at the largest uncertainties
        weight = np.diagonal(response_uncertainty_weight.magnitude, axis1=0, axis2=1)
        weight = weight / np.max(weight, axis=0)

    uncertainty_weight_list = []
    for idx_element in range(num_elements):
        # Extract the parameters relevant to each SISO uncertainty weight element
        magnitude_response_weight_element = np.array(
            response_uncertainty_weight.magnitude[idx_element, idx_element, :]
        )
        order_element = order_list[idx_element]
        weight_element = weight[:, idx_element]

        # Fit the uncertainty weight to each SISO element
        uncertainty_weight_element = utilities._fit_magnitude_log_chebyshev_siso(
            omega,
            magnitude_response_weight_element,
            order_element,
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
    frequency_response_nom: control.FrequencyResponseData,
    frequency_response_offnom_list: control.FrequencyResponseList,
):
    # System paramters
    num_inputs = frequency_response_nom.ninputs
    num_outputs = frequency_response_nom.noutputs

    # Initialize figure
    fig, ax = plt.subplots(
        num_outputs, num_inputs, sharex=True, layout="constrained", squeeze=False
    )

    # Off-nominal frequency response
    for frequency_response_offnom in frequency_response_offnom_list:
        frequency_offnom = frequency_response_offnom.frequency
        magnitude_offnom = frequency_response_offnom.magnitude
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].semilogx(
                    frequency_offnom,
                    control.mag2db(magnitude_offnom[idx_output, idx_input, :]),
                    color="tab:orange",
                    alpha=0.25,
                    label="Off-nominal",
                )

    # Nominal frequency response
    frequency_nom = frequency_response_nom.frequency
    magnitude_nom = frequency_response_nom.magnitude
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].semilogx(
                frequency_nom,
                control.mag2db(magnitude_nom[idx_output, idx_input, :]),
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
    frequency_response_nom: control.FrequencyResponseData,
    frequency_response_offnom_list: control.FrequencyResponseList,
):
    # System paramters
    num_inputs = frequency_response_nom.ninputs
    num_outputs = frequency_response_nom.noutputs

    # Initialize figure
    fig, ax = plt.subplots(
        num_outputs, num_inputs, sharex=True, layout="constrained", squeeze=False
    )

    # Off-nominal frequency response
    for frequency_response_offnom in frequency_response_offnom_list:
        frequency_offnom = frequency_response_offnom.frequency
        phase_offnom = frequency_response_offnom.phase
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].semilogx(
                    frequency_offnom,
                    180 / np.pi * phase_offnom[idx_output, idx_input, :],
                    color="tab:orange",
                    alpha=0.25,
                    label="Off-nominal",
                )

    # Nominal frequency response
    frequency_nom = frequency_response_nom.frequency
    phase_nom = frequency_response_nom.phase
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].semilogx(
                frequency_nom,
                180 / np.pi * phase_nom[idx_output, idx_input, :],
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
    frequency_response_nom: control.FrequencyResponseData,
    frequency_response_offnom_list: control.FrequencyResponseList,
):
    # Initialize figure
    fig, ax = plt.subplots(layout="constrained")

    # Off-nominal frequency response
    for frequency_response_offnom in frequency_response_offnom_list:
        frequency_offnom = frequency_response_offnom.frequency
        sval_offnom = np.linalg.svdvals(
            frequency_response_offnom.complex.transpose(2, 0, 1)
        )
        for idx_sval in range(sval_offnom.shape[1]):
            ax.semilogx(
                frequency_offnom,
                control.mag2db(sval_offnom[:, idx_sval]),
                color="tab:orange",
                alpha=0.5,
                label="Off-nominal",
            )

    # Nominal frequency response
    frequency_offnom = frequency_response_nom.frequency
    sval_nom = np.linalg.svdvals(frequency_response_nom.complex.transpose(2, 0, 1))
    for idx_sval in range(sval_offnom.shape[1]):
        ax.semilogx(
            frequency_offnom,
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
    uncertainty_residual_response_dict: Dict[str, control.FrequencyResponseList],
):
    for (
        uncertainty_model_id,
        uncertainty_residual_response_list,
    ) in uncertainty_residual_response_dict.items():
        frequency = uncertainty_residual_response_list[0].frequency
        fig, ax = plt.subplots()
        sval_response_residual_list = []
        for uncertainty_residual_response in uncertainty_residual_response_list:
            residual_response_matrix = uncertainty_residual_response.complex
            residual_response_sval = np.linalg.svdvals(
                residual_response_matrix.transpose(2, 0, 1)
            )
            sval_response_residual_list.append(residual_response_sval)
            for idx_sval in range(residual_response_sval.shape[1]):
                ax.semilogx(
                    frequency,
                    control.mag2db(residual_response_sval[:, idx_sval]),
                    color="grey",
                    alpha=0.5,
                    label=f"$\\sigma(E_{{{uncertainty_model_id}}})$",
                )
        sval_max_response_residual = np.max(
            np.array(sval_response_residual_list), axis=(0, 2)
        )
        ax.semilogx(
            frequency,
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
    uncertainty_residual_response_dict: Dict[str, control.FrequencyResponseList],
):
    # Maximum singular value response of uncertainty residuals
    sval_max_response_residual_dict = {}
    for (
        uncertainty_model_id,
        uncertainty_residual_response_list,
    ) in uncertainty_residual_response_dict.items():
        frequency = uncertainty_residual_response_list[0].frequency
        sval_response_residual_list = []
        for uncertainty_residual_response in uncertainty_residual_response_list:
            residual_response_matrix = uncertainty_residual_response.complex
            residual_response_sval = np.linalg.svdvals(
                residual_response_matrix.transpose(2, 0, 1)
            )
            sval_response_residual_list.append(residual_response_sval)
        sval_max_response_residual = np.max(
            np.array(sval_response_residual_list), axis=(0, 2)
        )
        sval_max_response_residual_dict[uncertainty_model_id] = (
            control.FrequencyResponseData(sval_max_response_residual, frequency)
        )

    # Initialize figure
    fig, ax = plt.subplots()

    # Maximum singular value reponse of uncertainty residuals
    for (
        uncertainty_model_id,
        sval_max_response_residual,
    ) in sval_max_response_residual_dict.items():
        frequency = sval_max_response_residual.frequency
        sval_max_response = sval_max_response_residual.magnitude
        ax.semilogx(
            frequency,
            control.mag2db(sval_max_response),
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
    response_weight_left: control.FrequencyResponseData,
    response_weight_right: control.FrequencyResponseData,
    weight_left: Optional[control.StateSpace] = None,
    weight_right: Optional[control.StateSpace] = None,
):
    #
    num_left = response_weight_left.ninputs
    num_right = response_weight_right.ninputs

    magnitude_response_weight_left = response_weight_left.magnitude
    magnitude_response_weight_right = response_weight_right.magnitude

    frequency = response_weight_left.frequency

    # Initialize figure
    fig, ax = plt.subplots(
        max(num_left, num_right), 2, sharex=True, layout="constrained"
    )

    # Plot left uncertainty weight frequency response
    for idx_left in range(num_left):
        ax[idx_left, 0].semilogx(
            frequency,
            control.mag2db(magnitude_response_weight_left[idx_left, idx_left, :]),
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
            frequency,
            control.mag2db(magnitude_response_weight_right[idx_right, idx_right, :]),
            linestyle="",
            marker="*",
            color="tab:blue",
            label="Response",
        )
        ax[idx_right, 1].set_ylabel("$|W_{R, (1, 1)}|$ (dB)")
        ax[idx_right, 1].grid()

    # Plot left uncertainty weight fit frequency response
    if weight_left is not None:
        response_fit_weight_left = control.frequency_response(weight_left, frequency)
        magnitude_response_fit_weight_left = response_fit_weight_left.magnitude
        for idx_left in range(num_left):
            ax[idx_left, 0].semilogx(
                frequency,
                control.mag2db(
                    magnitude_response_fit_weight_left[idx_left, idx_left, :]
                ),
                color="tab:orange",
                label="Fit",
            )
    # Plot right uncertainty weight fit frequency response
    if weight_right is not None:
        response_fit_weight_right = control.frequency_response(weight_right, frequency)
        magnitude_response_fit_weight_right = response_fit_weight_right.magnitude
        for idx_right in range(num_right):
            ax[idx_right, 1].semilogx(
                frequency,
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
