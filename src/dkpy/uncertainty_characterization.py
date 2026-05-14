"""Uncertainty characterization utilties."""

__all__ = [
    "compute_uncertainty_residual_response",
    "compute_uncertainty_weight_response",
    "compute_uncertainty_measure_response",
    "fit_uncertainty_weight",
    "plot_magnitude_response_uncertain_model_set",
    "plot_phase_response_uncertain_model_set",
    "plot_singular_value_response_uncertain_model_set",
    "plot_singular_value_response_residual",
    "plot_singular_value_response_residual_comparison",
    "plot_singular_value_response_uncertainty_weight",
]

import warnings

import control
import numpy as np
import cvxpy
import scipy
from matplotlib import pyplot as plt

from typing import List, Optional, Union, Tuple, Dict, Any, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend

from . import lti_system_fit

UncertaintyModelId = Literal[
    "additive",
    "multiplicative_input",
    "multiplicative_output",
    "inverse_additive",
    "inverse_multiplicative_input",
    "inverse_multiplicative_output",
]


def compute_uncertainty_residual_response(
    complex_response_nom: Union[np.ndarray, control.FrequencyResponseData],
    complex_response_offnom_list: Union[np.ndarray, control.FrequencyResponseList],
    uncertainty_model: Union[UncertaintyModelId, List[UncertaintyModelId]],
    tol_residual_existence: float = 1e-12,
) -> Dict[UncertaintyModelId, np.ndarray]:
    """Compute the residual response of unstructured uncertainty models.

    Parameters
    ----------
    complex_response_nom : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response of the nominal system.
    complex_response_offnom_list : Union[np.ndarray, control.FrequencyResponseList]
        Frequency response of the off-nominal system.
    uncertainty_model : Union[UncertaintyModelId, List[UncertaintyModelId]]
        Uncertainty model identifiers to compute the residual response.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    Dict[str, np.ndarray]
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
    >>> uncertainty_models = {
    ...     "additive",
    ...     "multiplicative_input",
    ...     "multiplicative_output",
    ...     "inverse_additive",
    ...     "inverse_multiplicative_input",
    ...     "inverse_multiplicative_output",
    ... }
    >>> complex_response_residuals_dict = dkpy.compute_uncertainty_residual_response(
    ...     complex_response_nom,
    ...     complex_response_offnom_list,
    ...     uncertainty_models,
    ... )
    """

    # Convert frequency response data to expected type
    complex_response_nom = _convert_frequency_response_data_to_array(
        complex_response_nom
    )
    complex_response_offnom_list = _convert_frequency_response_list_to_array(
        complex_response_offnom_list
    )

    # Uncertainty residual response dictionary
    complex_response_residual_dict = {}

    compute_residual_dispatcher = {
        "additive": _compute_residual_additive,
        "multiplicative_input": _compute_residual_multiplicative_input,
        "multiplicative_output": _compute_residual_multiplicative_output,
        "inverse_additive": _compute_residual_inverse_additive,
        "inverse_multiplicative_input": _compute_residual_inverse_multiplicative_input,
        "inverse_multiplicative_output": _compute_residual_inverse_multiplicative_output,
    }

    for model in uncertainty_model:
        try:
            compute_residual_model = compute_residual_dispatcher[model]
        except KeyError:
            raise KeyError(
                "The uncertainty model identifier must be `additive`, "
                "`multiplicative_input`, `multiplicative_output`, `inverse_additive` "
                "`inverse_multiplicative_input`, or `inverse_multiplicative_output` "
                f"(got `{model}`)."
            )
        complex_response_residual_list = compute_residual_model(
            complex_response_nom,
            complex_response_offnom_list,
            tol_residual_existence,
        )
        complex_response_residual_dict[model] = complex_response_residual_list

    return complex_response_residual_dict


def _compute_residual_additive(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: Optional[float] = None,
) -> np.ndarray:
    """Compute the additive uncertainty residual frequency responses.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model frequency response.
    complex_offnominal : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Additive uncertainty residual frequency response for all off-nominal models.
    """
    complex_residual = complex_offnominal - complex_nominal

    return complex_residual


def _compute_residual_multiplicative_input(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: float = 1e-8,
) -> np.ndarray:
    """Compute the multiplicative input uncertainty residual frequency responses.

    Parameters
    ----------
    complex_response_nom : np.ndarray
        Nominal model frequency response.
    complex_response_offnom : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative input uncertainty residual frequency response for all
        off-nominal models.

    Raises
    ------
    ValueError
        An input multiplicative uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    nbr_inputs = complex_nominal.shape[-1]
    nbr_outputs = complex_nominal.shape[-2]

    a = complex_nominal
    b = complex_offnominal - complex_nominal
    x, residues_lstsq, _, _ = scipy.linalg.lstsq(a, b)
    complex_residual = x

    if nbr_inputs >= nbr_outputs:
        return complex_residual
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_residual
        else:
            raise ValueError(
                "A multiplicative input uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is due "
                "to the fact that the multiplicative input uncertainty model cannot "
                "account for all off-nominal models when the number of inputs is less "
                "than the number of outputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_residual_multiplicative_output(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: float = 1e-8,
) -> np.ndarray:
    """Compute the multiplicative output uncertainty residual frequency responses.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model frequency response.
    complex_offnominal : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Multiplicative output uncertainty residual frequency response for all
        off-nominal models.

    Raises
    ------
    ValueError
        An output multiplicative uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    nbr_inputs = complex_nominal.shape[-1]
    nbr_outputs = complex_nominal.shape[-2]

    a = np.moveaxis(complex_nominal, -1, -2)
    b = np.moveaxis(complex_offnominal - complex_nominal, -1, -2)
    x, residues_lstsq, _, _ = scipy.linalg.lstsq(a, b)
    complex_residual = np.moveaxis(x, -1, -2)

    if nbr_inputs <= nbr_outputs:
        return complex_residual
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_residual
        else:
            raise ValueError(
                "A multiplicative output uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is "
                "due to the fact that the multiplicative output uncertainty model "
                "cannot account for all off-nominal models when the number of outputs "
                "is less than the number of inputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_residual_inverse_additive(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: float = 1e-8,
) -> np.ndarray:
    """Compute the inverse additive uncertainty residual frequency responses.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model frequency response.
    complex_offnominal : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse additive uncertainty residual frequency response for all
        off-nominal models.

    Raises
    ------
    ValueError
        An inverse additive uncertainty residual does not exist at the given
        frequency as the linear system used to compute the residual does not have a
        solution.
    """

    nbr_inputs = complex_nominal.shape[-1]
    nbr_outputs = complex_nominal.shape[-2]

    a1 = complex_offnominal
    b1 = complex_offnominal - complex_nominal
    y, residues_lstsq_1, _, _ = scipy.linalg.lstsq(a1, b1)
    a2 = np.moveaxis(complex_nominal, -1, -2)
    b2 = np.moveaxis(y, -1, -2)
    x, residues_lstsq_2, _, _ = scipy.linalg.lstsq(a2, b2)
    complex_residual = np.moveaxis(x, -1, -2)

    if nbr_inputs == nbr_outputs:
        return complex_residual
    else:
        if np.all(residues_lstsq_1 <= tol_residual_existence) and np.all(
            residues_lstsq_2 <= tol_residual_existence
        ):
            return complex_residual
        else:
            raise ValueError(
                "An inverse additive uncertainty residual does not exist for the "
                "given nominal and off-nominal frequency response matrix. This is "
                "due to the fact that the inverse additive uncertainty model "
                "cannot account for all off-nominal models when the number of inputs "
                "not equal to the number of outputs. Please consider using a different "
                "uncertainty model."
            )


def _compute_residual_inverse_multiplicative_input(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: float = 1e-8,
) -> np.ndarray:
    """Compute the inverse multiplicative input uncertainty residual frequency responses.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model frequency response.
    complex_offnominal : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative input uncertainty residual frequency response for all
        off-nominal models.

    Raises
    ------
    ValueError
        An inverse multiplicative input uncertainty residual does not exist at the
        given frequency as the linear system used to compute the residual does not have
        a solution.
    """

    nbr_inputs = complex_nominal.shape[-1]
    nbr_outputs = complex_nominal.shape[-2]

    a = complex_offnominal
    b = complex_offnominal - complex_nominal
    x, residues_lstsq, _, _ = scipy.linalg.lstsq(a, b)
    complex_residual = x

    if nbr_inputs >= nbr_outputs:
        return complex_residual
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_residual
        else:
            raise ValueError(
                "An inverse multiplicative input uncertainty residual does not exist "
                "for the given nominal and off-nominal frequency response matrix. This "
                "is due to the fact that the inverse multiplicative input uncertainty "
                "model cannot account for all off-nominal models when the number of "
                "inputs is less than the number of outputs. Please consider using a "
                "different uncertainty model."
            )


def _compute_residual_inverse_multiplicative_output(
    complex_nominal: np.ndarray,
    complex_offnominal: np.ndarray,
    tol_residual_existence: float = 1e-8,
) -> np.ndarray:
    """Compute the inverse multiplicative output uncertainty residual frequency responses.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model frequency response.
    complex_offnominal : np.ndarray
        Off-nominal model frequency responses.
    tol_residual_existence : float
        Tolerance for the existence of an uncertainty residual.

    Returns
    -------
    np.ndarray
        Inverse multiplicative output uncertainty residual frequency response for all
        off-nominal models.

    Raises
    ------
    ValueError
        An inverse multiplicative output uncertainty residual does not exist at the
        given frequency as the linear system used to compute the residual does not have
        a solution.
    """

    nbr_inputs = complex_nominal.shape[-1]
    nbr_outputs = complex_nominal.shape[-2]

    a = np.moveaxis(complex_offnominal, -1, -2)
    b = np.moveaxis(complex_offnominal - complex_nominal, -1, -2)
    x, residues_lstsq, _, _ = scipy.linalg.lstsq(a, b)
    complex_residual = np.moveaxis(x, -1, -2)

    if nbr_inputs <= nbr_outputs:
        return complex_residual
    else:
        if np.all(residues_lstsq <= tol_residual_existence):
            return complex_residual
        else:
            raise ValueError(
                "An inverse multiplicative output uncertainty residual does not exist "
                "for the given nominal and off-nominal frequency response matrix. This "
                "is due to the fact that the inverse multiplicative output uncertainty "
                "model cannot account for all off-nominal models when the number of "
                "outputs is less than the number of inputs. Please consider using a "
                "different uncertainty model."
            )


def compute_uncertainty_weight_response(
    complex_residual: Union[np.ndarray, control.FrequencyResponseList],
    weight_left_structure: Literal["full", "diagonal", "scalar", "identity"],
    weight_right_structure: Literal["full", "diagonal", "scalar", "identity"],
    solver_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal uncertainty weight frequency response.

    The algorithm is based on a modified semidefinite program formulation presented in
    [#uncertainty_characterization]_.

    Parameters
    ----------
    complex_residual : Union[np.ndarray, control.FrequencyResponseList]
        Frequency response of the residuals for which to compute the optimal uncertainty
        weights.
    weight_left_structure : Literal["full", "diagonal", "scalar", "identity"]
        Structure of the left uncertainty weight.
    weight_right_structure : Literal["full", "diagonal", "scalar", "identity"]
        Structure of the right uncertainty weight.
    solver_params : Dict[str, Any]
        Keyword arguments for the convex optimization solver. See [#cvxpy_solver]_ for
        more information.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency response of the diagonal elements of the left and right uncertainty
        weights.

    Examples
    --------
    Compute the optimal left and right uncertainty weights for the multiplicative input
    residual frequency response where the left and right uncertainty weights are assumed
    to be diagonal.

    >>> complex_nominal, complex_offnominal_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_models = [
    ...     "additive",
    ...     "multiplicative_input",
    ...     "multiplicative_output",
    ...     "inverse_additive",
    ...     "inverse_multiplicative_input",
    ...     "inverse_multiplicative_output",
    ... ]
    >>> complex_residual_dict = compute_uncertainty_residual_response(
    ...     complex_nominal,
    ...     complex_offnominal_list,
    ...     uncertainty_models,
    ... )
    >>> complex_weight_left, complex_weight_right = (
    ...     dkpy.compute_uncertainty_weight_response(
    ...         complex_residual_dict["multiplicative_input"],
    ...         "diagonal",
    ...         "diagonal",
    ...     )
    ... )

    References
    ----------
    .. [#uncertainty_characterization] G. J. Balas, A. K. Packard, and P. J. Seiler,
    “Uncertain Model Set Calculation from Frequency Domain Data,” Springer eBooks,
    pp. 89–105, Jan. 2009, doi: https://doi.org/10.1007/978-1-4419-0895-7_6.
    """

    # Convert frequency response data to expected type
    complex_residual = _convert_frequency_response_list_to_array(complex_residual)

    # Parse solver parameters
    solver_params = (
        {
            "solver": cvxpy.CLARABEL,
            "tol_gap_abs": 1e-6,
            "tol_gap_rel": 1e-6,
            "tol_feas": 1e-6,
            "tol_infeas_abs": 1e-6,
            "tol_infeas_rel": 1e-6,
        }
        if solver_params is None
        else solver_params
    )

    # Frequency response parameters
    nbr_frequency = complex_residual.shape[1]
    nbr_left = complex_residual.shape[2]
    nbr_right = complex_residual.shape[3]
    nbr_offnom = complex_residual.shape[0]

    # Generate left weight variable
    weight_left_power_dispatcher = {
        "full": cvxpy.Variable((nbr_left, nbr_left), hermitian=True),
        "diagonal": cvxpy.Variable((nbr_left, nbr_left), diag=True),
        "scalar": cvxpy.Variable() * scipy.sparse.eye_array(nbr_left),
        "identity": cvxpy.Constant(value=np.eye(nbr_left)),
    }
    try:
        weight_left_power = weight_left_power_dispatcher[weight_left_structure]
    except KeyError:
        raise KeyError(
            '`weight_left_structure` must be "full", "diagonal", or "scalar" (got '
            f'"{weight_left_structure})".'
        )

    # Generate right weight variable
    weight_right_power_dispatcher = {
        "full": cvxpy.Variable((nbr_right, nbr_right), hermitian=True),
        "diagonal": cvxpy.Variable((nbr_right, nbr_right), diag=True),
        "scalar": cvxpy.Variable() * scipy.sparse.eye_array(nbr_right),
        "identity": cvxpy.Constant(value=np.eye(nbr_right)),
    }
    try:
        weight_right_power = weight_right_power_dispatcher[weight_right_structure]
    except KeyError:
        raise KeyError(
            '`weight_right_structure` must be "full", "diagonal", or "scalar" (got '
            f'"{weight_right_structure})".'
        )

    # Generate residual parameters
    residual_offnom = cvxpy.Parameter(
        shape=(nbr_offnom, nbr_left, nbr_right),
        complex=True,
    )

    # Uncertainty set constraints over all frequencies
    constraint_list = []
    for idx_offnom in range(nbr_offnom):
        constraint_matrix_freq = cvxpy.bmat(
            [
                [weight_left_power, residual_offnom[idx_offnom, :, :]],
                [residual_offnom[idx_offnom, :, :].H, weight_right_power],
            ]
        )
        constraint_freq = constraint_matrix_freq >> 0
        constraint_list.append(constraint_freq)

    # Positive semidefiniteness constraints
    constraint_list.append(weight_left_power >> 0)
    constraint_list.append(weight_right_power >> 0)

    # Semidefinite program
    objective = cvxpy.Minimize(
        nbr_right * cvxpy.trace(weight_left_power)
        + nbr_left * cvxpy.trace(weight_right_power)
    )
    problem = cvxpy.Problem(objective, constraint_list)

    # Compute optimal uncertainty weights
    complex_weight_left = []
    complex_weight_right = []
    for idx_freq in range(nbr_frequency):
        # Residual of off-nominal models at given frequency
        residual_offnom.value = complex_residual[:, idx_freq, :, :]

        # Solve optimal weight SDP
        problem.solve(canon_backend=cvxpy.SCIPY_CANON_BACKEND, **solver_params)

        # Extract left weight
        if weight_left_structure == "identity" or weight_left_structure == "full":
            weight_left_power_opt = np.array(weight_left_power.value)
        else:
            weight_left_power_opt = np.array(weight_left_power.value.toarray())
        weight_left_opt = scipy.linalg.sqrtm(weight_left_power_opt)

        # Extract right weight
        if weight_right_structure == "identity" or weight_right_structure == "full":
            weight_right_power_opt = np.array(weight_right_power.value)
        else:
            weight_right_power_opt = np.array(weight_right_power.value.toarray())
        weight_right_opt = scipy.linalg.sqrtm(weight_right_power_opt)

        complex_weight_left.append(weight_left_opt)
        complex_weight_right.append(weight_right_opt)

    # Generate uncertainty weight complex frequency reponses
    complex_weight_left = np.array(complex_weight_left)
    complex_weight_right = np.array(complex_weight_right)

    return complex_weight_left, complex_weight_right


def fit_uncertainty_weight(
    complex_uncertainty_weight: Union[np.ndarray, control.FrequencyResponseData],
    omega: np.ndarray,
    order: Union[int, List[int], np.ndarray],
    uncertainty_weight_type: Literal["left", "right"],
    uncertainty_weight_structure: Literal["scalar", "diagonal", "full"],
    weight: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
) -> control.StateSpace:
    """Fit an overbounding stable and minimum-phase uncertainty weight.

    Parameters
    ----------
    complex_uncertainty_weight : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response of uncertainty weight.
    order : Union[int, List[int], np.ndarray]
        Order of the uncertainty weight model.
    uncertainty_weight_type: Literal["left", "right"],
        Identifier for the left or right uncertainty weight.
    uncertainty_weight_structure : Literal["scalar", "diagonal", "full"]
        Structure constraint for the uncertainty weight.
    weight : Optional[np.ndrray] = None
        Frequency-dependent weight for fit accuracy.
    solver_params: Optional[Dict[str, Any]]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum number of iterations for the bisection algorithm.
    max_iter_bisection_init : int
        Maximum number of iterations for the bisection algorithm initialization.
    nbr_power_constraint : Optional[np.ndarray]
        Number of frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    control.StateSpace
        Overbounding stable and minimum-phase uncertainty weight.

    Examples
    --------
    Fit an overbounding stable and minimum phase system to the left and right
    uncertainty weights for the multiplicative input uncertainty model.

    >>> complex_response_nom, complex_response_offnom_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_models = ["multiplicative_input"]
    >>> complex_response_residual_dict = compute_uncertainty_residual_response(
    ...     complex_response_nom,
    ...     complex_response_offnom_list,
    ...     uncertainty_models,
    ... )
    >>> complex_response_weight_left, complex_response_weight_right = (
    ...     dkpy.compute_uncertainty_weight_response(
    ...         complex_response_residual_dict["multiplicative_input"],
    ...         "diagonal",
    ...         "diagonal",
    ...     )
    ... )
    >>> weight_left = dkpy.fit_uncertainty_weight(
    ...     complex_response_weight_left, omega, [4, 5], "left", "diagonal"
    ... )
    >>> weight_right = dkpy.fit_uncertainty_weight(
    ...     complex_response_weight_right, omega, [3, 5], "right", "diagonal"
    ... )

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Convert frequency response data to expected type
    complex_uncertainty_weight = _convert_frequency_response_data_to_array(
        complex_uncertainty_weight
    )

    # Solver settings
    solver_params = (
        {
            "solver": cvxpy.CLARABEL,
            "tol_gap_abs": 1e-9,
            "tol_gap_rel": 1e-9,
            "tol_feas": 1e-9,
            "tol_infeas_abs": 1e-9,
            "tol_infeas_rel": 1e-9,
        }
        if solver_params is None
        else solver_params
    )

    # Fit method dispatcher
    fit_method_dispatcher = {
        "scalar": _fit_uncertainty_weight_scalar,
        "diagonal": _fit_uncertainty_weight_diagonal,
        "full": _fit_uncertainty_weight_full,
    }

    # Select uncertainty weight fit method based on weight structure
    try:
        fit_uncertainty_weight_method = fit_method_dispatcher[
            uncertainty_weight_structure
        ]
    except KeyError:
        raise KeyError(
            "Exptected `weight_structure` to be `scalar`, `diagonal` or `full` (got "
            f"`{uncertainty_weight_structure}`)."
        )

    # Fit uncertainty weight
    uncertainty_weight = fit_uncertainty_weight_method(
        complex_uncertainty_weight,
        omega,
        order,
        uncertainty_weight_type,
        weight,
        solver_params,
        tol_bisection,
        max_iter_bisection,
        max_iter_bisection_init,
        nbr_power_constraint,
    )

    return uncertainty_weight


def _fit_uncertainty_weight_scalar(
    complex_uncertainty_weight: np.ndarray,
    omega: np.ndarray,
    order: int,
    uncertainty_weight_type: Literal["left", "right"],
    weight: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
) -> control.StateSpace:
    """Fit an overbounding stable and minimum-phase scalar uncertainty weight.

    Parameters
    ----------
    complex_uncertainty_weight : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response of uncertainty weight.
    order : Union[int, List[int], np.ndarray]
        Order of the uncertainty weight model.
    uncertainty_weight_type: Literal["left", "right"],
        Identifier for the left or right uncertainty weight.
    weight : Optional[np.ndrray] = None
        Frequency-dependent weight for fit accuracy.
    solver_params: Optional[Dict[str, Any]]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum number of iterations for the bisection algorithm.
    max_iter_bisection_init : int
        Maximum number of iterations for the bisection algorithm initialization.
    nbr_power_constraint : Optional[np.ndarray]
        Number of frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    control.StateSpace
        Overbounding stable and minimum-phase uncertainty weight.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Auxiliary parameters
    nbr_signals = complex_uncertainty_weight.shape[1]

    # Compute the magnitude of the scalar weight. Given that a scalar weight is assumed,
    # the weight contains the same values on the diagonals and zeros elsewhere.
    # Therefore, we take the first diagonal as they are identical along the diagonal.
    magnitude_uncertainty_weight = np.abs(complex_uncertainty_weight[:, 0, 0])

    # Fit the scalar uncertainty weight
    uncertainty_weight_scalar = lti_system_fit.fit_magnitude_siso_ct(
        magnitude_fit=magnitude_uncertainty_weight,
        omega=omega,
        order=order,
        magnitude_upper_bound=None,
        magnitude_lower_bound=magnitude_uncertainty_weight,
        weight=weight,
        solver_params=solver_params,
        tol_bisection=tol_bisection,
        max_iter_bisection=max_iter_bisection,
        max_iter_bisection_init=max_iter_bisection_init,
        nbr_power_constraint=nbr_power_constraint,
    )

    # Duplicate the scalar uncertainty weight along the diagonal
    uncertainty_weight = control.append(*([uncertainty_weight_scalar] * nbr_signals))

    return uncertainty_weight


def _fit_uncertainty_weight_diagonal(
    complex_uncertainty_weight: np.ndarray,
    omega: np.ndarray,
    order: Union[int, List[int]],
    uncertainty_weight_type: Literal["left", "right"],
    weight: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
):
    """Fit an overbounding stable and minimum-phase diagonal uncertainty weight.

    Parameters
    ----------
    complex_uncertainty_weight : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response of uncertainty weight.
    order : Union[int, List[int], np.ndarray]
        Order of the uncertainty weight model.
    uncertainty_weight_type: Literal["left", "right"],
        Identifier for the left or right uncertainty weight.
    weight : Optional[np.ndrray] = None
        Frequency-dependent weight for fit accuracy.
    solver_params: Optional[Dict[str, Any]]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum number of iterations for the bisection algorithm.
    max_iter_bisection_init : int
        Maximum number of iterations for the bisection algorithm initialization.
    nbr_power_constraint : Optional[np.ndarray]
        Number of frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    control.StateSpace
        Overbounding stable and minimum-phase uncertainty weight.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """
    # Auxiliary parameters
    nbr_signals = complex_uncertainty_weight.shape[1]

    # Parse order into list format
    order_list = [order] * nbr_signals if isinstance(order, int) else order

    # Fit the diagonal uncertainty weight elements
    uncertainty_weight_list = []
    for idx in range(nbr_signals):
        # Compute the magnitude of the diagonal uncertainty weight element
        magnitude_uncertainty_weight = np.abs(complex_uncertainty_weight[:, idx, idx])
        order_idx = order_list[idx]

        # Fit the diagonal uncertainty weight element
        uncertainty_weight_diagonal = lti_system_fit.fit_magnitude_siso_ct(
            magnitude_fit=magnitude_uncertainty_weight,
            omega=omega,
            order=order_idx,
            magnitude_upper_bound=None,
            magnitude_lower_bound=magnitude_uncertainty_weight,
            weight=weight,
            solver_params=solver_params,
            tol_bisection=tol_bisection,
            max_iter_bisection=max_iter_bisection,
            max_iter_bisection_init=max_iter_bisection_init,
            nbr_power_constraint=nbr_power_constraint,
        )
        uncertainty_weight_list.append(uncertainty_weight_diagonal)

    # Construct the diagonal uncertainty weight from the diagonal elements
    uncertainty_weight = control.append(*uncertainty_weight_list)

    return uncertainty_weight


def _fit_uncertainty_weight_full(
    complex_uncertainty_weight: np.ndarray,
    omega: np.ndarray,
    order: int,
    uncertainty_weight_type: Literal["left", "right"],
    weight: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
):
    """Fit an overbounding stable and minimum-phase full uncertainty weight.

    Parameters
    ----------
    complex_uncertainty_weight : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response of uncertainty weight.
    order : Union[int, List[int], np.ndarray]
        Order of the uncertainty weight model.
    uncertainty_weight_type: Literal["left", "right"],
        Identifier for the left or right uncertainty weight.
    weight : Optional[np.ndrray] = None
        Frequency-dependent weight for fit accuracy.
    solver_params: Optional[Dict[str, Any]]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum number of iterations for the bisection algorithm.
    max_iter_bisection_init : int
        Maximum number of iterations for the bisection algorithm initialization.
    nbr_power_constraint : Optional[np.ndarray]
        Number of frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    control.StateSpace
        Overbounding stable and minimum-phase uncertainty weight.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    raise NotImplementedError()


def compute_uncertainty_measure_response(
    complex_nominal: Union[np.ndarray, control.FrequencyResponseData],
    complex_weight_left: Union[np.ndarray, control.FrequencyResponseData],
    complex_weight_right: Union[np.ndarray, control.FrequencyResponseData],
    uncertainty_model: UncertaintyModelId,
) -> np.ndarray:
    """Compute measure (size/volume) frequency response of uncertainty model.

    Parameters
    ----------
    complex_nominal : Union[np.ndarray, control.FrequencyResponseData]
        Nominal model complex frequency response.
    complex_weight_left : Union[np.ndarray, control.FrequencyResponseData]
        Left uncertainty weight complex frequency response.
    complex_weight_right : Union[np.ndarray, control.FrequencyResponseData]
        Right uncertainty weight complex frequency reponse.
    uncertainty_model : UncertaintyModelId
        Uncertainty model identifier.

    Returns
    -------
    np.ndarray
        Measure frequency response.


    Examples
    --------
    Compute the uncertainty measure response for a multiplicative input uncertainty
    model.

    >>> complex_nominal, complex_offnominal_list, omega = (
    ...     example_multimodel_uncertainty
    ... )
    >>> uncertainty_models = ["multiplicative_input"]
    >>> complex_residual_dict = compute_uncertainty_residual_response(
    ...     complex_nominal,
    ...     complex_offnominal_list,
    ...     uncertainty_models,
    ... )
    >>> complex_weight_left, complex_weight_right = (
    ...     dkpy.compute_uncertainty_weight_response(
    ...         complex_residual_dict["multiplicative_input"],
    ...         "diagonal",
    ...         "diagonal",
    ...     )
    ... )
    >>> measure = dkpy.compute_uncertainty_measure_response(
    ...     complex_nominal,
    ...     complex_weight_left,
    ...     complex_weight_right,
    ...     "multiplicative_input",
    ... )
    """

    complex_nominal = _convert_frequency_response_data_to_array(complex_nominal)
    complex_weight_left = _convert_frequency_response_data_to_array(complex_weight_left)
    complex_weight_right = _convert_frequency_response_data_to_array(
        complex_weight_right
    )

    transform_to_additive_dispatcher = {
        "additive": _transform_additive_to_additive,
        "multiplicative_input": _transform_multiplicative_input_to_additive,
        "multiplicative_output": _transform_multiplicative_output_to_additive,
        "inverse_additive": _transform_inverse_additive_to_additive,
        "inverse_multiplicative_input": _transform_inverse_multiplicative_input_to_additive,
        "inverse_multiplicative_output": _transform_inverse_multiplicative_output_to_additive,
    }

    # Compute equivalent additive uncertainty model frequency response
    complex_uncertainty_add = transform_to_additive_dispatcher[uncertainty_model](
        complex_nominal, complex_weight_left, complex_weight_right
    )
    complex_nominal_add = complex_uncertainty_add[0]
    complex_weight_left_add = complex_uncertainty_add[1]
    complex_weight_right_add = complex_uncertainty_add[2]

    # Compute minimal representation of additive uncertainty frequency response
    complex_weight_add_min = _compute_minimal_weights_additive(
        complex_weight_left_add, complex_weight_right_add
    )
    complex_weight_left_add = complex_weight_add_min[0]
    complex_weight_right_add = complex_weight_add_min[1]

    # Compute measure frequency response of minimal additive uncertainty response
    measure = _compute_uncertainty_measure_additive(
        complex_weight_left_add, complex_weight_right_add
    )

    return measure


def _compute_uncertainty_measure_additive(
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> np.ndarray:
    """Compute measure (size/volume) frequency response of additive uncertainty.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of additive uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of additive uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of additive uncertainty

    Returns
    -------
    np.ndarray
        Measure frequency response of additive uncertainty.

    Raises
    ------
    ValueError
        Left uncertainty weight is non-square.
    ValueError
        Right uncertainty weight is non-square.
    """

    # Check uncertainty weight dimensions
    if complex_weight_left.shape[1] != complex_weight_left.shape[2]:
        raise ValueError(
            "Left uncertainty weight must be square (got "
            f"{complex_weight_left.shape[1]} rows and {complex_weight_left.shape[2]} "
            f"columns)."
        )
    if complex_weight_right.shape[1] != complex_weight_right.shape[2]:
        raise ValueError(
            "Right uncertainty weight must be square (got "
            f"{complex_weight_right.shape[1]} rows and {complex_weight_right.shape[2]} "
            f"columns)."
        )

    # Auxiliary parameters
    nbr_inputs = complex_weight_right.shape[1]
    nbr_outputs = complex_weight_left.shape[1]

    # Measure computation
    measure_weight_left = np.abs(np.linalg.det(complex_weight_left)) ** (
        2 / nbr_outputs
    )
    measure_weight_right = np.abs(np.linalg.det(complex_weight_right)) ** (
        2 / nbr_inputs
    )
    measure = measure_weight_left * measure_weight_right

    return measure


def _transform_additive_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform additive uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of additive uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of additive uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of additive uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """

    # Nominal model
    complex_nominal_add = complex_nominal

    # Left uncertainty weight
    complex_weight_left_add = complex_weight_left

    # Right uncertainty weight
    complex_weight_right_add = complex_weight_right

    return complex_nominal_add, complex_weight_left_add, complex_weight_right_add


def _transform_multiplicative_input_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform multiplicative input uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of multiplicative input uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of multiplicative input
        uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of multiplicative input
        uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """

    # Nominal model
    complexnominal_add = complex_nominal

    # Left uncertainty weight
    complex_weight_left_add = complex_nominal @ complex_weight_left

    # Right uncertainty weight
    complex_weight_right_add = complex_weight_right

    return complexnominal_add, complex_weight_left_add, complex_weight_right_add


def _transform_multiplicative_output_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform multiplicative output uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of multiplicative output uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of multiplicative output
        uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of multiplicative output
        uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """

    # Nominal model
    complex_nominal_add = complex_nominal

    # Left uncertainty weight
    complex_weight_left_add = complex_weight_left

    # Right uncertainty weight
    complex_weight_right_add = complex_weight_right @ complex_nominal

    return complex_nominal_add, complex_weight_left_add, complex_weight_right_add


def _transform_inverse_additive_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform inverse additive uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of inverse additive uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of inverse additive
        uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of inverse additive
        uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """

    # Auxiliary variables
    product = complex_weight_right @ complex_nominal @ complex_weight_left
    product_herm = np.moveaxis(product.conj(), -1, -2)
    eye_outer = np.eye(product.shape[1])[None, :, :]
    eye_inner = np.eye(product.shape[2])[None, :, :]

    # Nominal model
    a_nominal = eye_outer - product @ product_herm
    b_nominal = complex_weight_right @ complex_nominal
    x_nominal = np.linalg.solve(a_nominal, b_nominal)
    complex_nominal_add = (
        complex_nominal
        + complex_nominal @ complex_weight_left @ product_herm @ x_nominal
    )

    # Left uncertainty weight
    # The transpose using np.moveaxis is required as the linear system solved is in the
    # form X @ A = B whereas numpy requires C @ X = D. The original problem is converted
    # to the equivalent problem A.T @ X.T = B.T.
    a_left = np.moveaxis(scipy.linalg.sqrtm(eye_inner - product_herm @ product), -1, -2)
    b_left = np.moveaxis(-(complex_nominal @ complex_weight_left), -1, -2)
    complex_weight_left_add = np.moveaxis(np.linalg.solve(a_left, b_left), -1, -2)

    # Right uncertainty weight
    a_right = scipy.linalg.sqrtm(eye_outer - product @ product_herm)
    b_right = complex_weight_right @ complex_nominal
    complex_weight_right_add = np.linalg.solve(a_right, b_right)

    return complex_nominal_add, complex_weight_left_add, complex_weight_right_add


def _transform_inverse_multiplicative_input_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform inverse multiplicative input uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of inverse multiplicative input
        uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of inverse multiplicative
        input uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of inverse multiplicative
        input uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """
    # Auxiliary variables
    product = complex_weight_right @ complex_weight_left
    product_herm = np.moveaxis(product.conj(), -1, -2)
    eye_outer = np.eye(product.shape[1])[None, :, :]
    eye_inner = np.eye(product.shape[2])[None, :, :]

    # Nominal model
    a_nominal = eye_outer - product @ product_herm
    b_nominal = complex_weight_right
    x_nominal = np.linalg.solve(a_nominal, b_nominal)
    complex_nominal_add = (
        complex_nominal
        + complex_nominal @ complex_weight_left @ product_herm @ x_nominal
    )

    # Left uncertainty weight
    # The transpose using np.moveaxis is required as the linear system solved is in the
    # form X @ A = B whereas numpy requires C @ X = D. The original problem is converted
    # to the equivalent problem A.T @ X.T = B.T.
    a_left = np.moveaxis(scipy.linalg.sqrtm(eye_inner - product_herm @ product), -1, -2)
    b_left = np.moveaxis(-(complex_nominal @ complex_weight_left), -1, -2)
    complex_weight_left_add = np.moveaxis(np.linalg.solve(a_left, b_left), -1, -2)

    # Right uncertainty weight
    a_right = scipy.linalg.sqrtm(eye_outer - product @ product_herm)
    b_right = complex_weight_right
    complex_weight_right_add = np.linalg.solve(a_right, b_right)

    return complex_nominal_add, complex_weight_left_add, complex_weight_right_add


def _transform_inverse_multiplicative_output_to_additive(
    complex_nominal: np.ndarray,
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform inverse multiplicative output uncertainty frequency response to additive form.

    Parameters
    ----------
    complex_nominal : np.ndarray
        Nominal model complex frequency response of inverse multiplicative output
        uncertainty.
    complex_weight_left : np.ndarray
        Left uncertainty weight complex frequency response of inverse multiplicative
        output uncertainty.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex frequency response of inverse multiplicative
        output uncertainty.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Nominal model, left uncertainty weight, and right uncertainty weight complex
        frequency response of equivalent additive uncertainty.
    """
    # Auxiliary variables
    product = complex_weight_right @ complex_weight_left
    product_herm = np.moveaxis(product.conj(), -1, -2)
    eye_outer = np.eye(product.shape[1])[None, :, :]
    eye_inner = np.eye(product.shape[2])[None, :, :]

    # Nominal model
    a_nominal = eye_outer - product @ product_herm
    b_nominal = complex_weight_right @ complex_nominal
    x_nominal = np.linalg.solve(a_nominal, b_nominal)
    complex_nominal_add = (
        complex_nominal + complex_weight_left @ product_herm @ x_nominal
    )

    # Left uncertainty weight
    # The transpose using np.moveaxis is required as the linear system solved is in the
    # form X @ A = B whereas numpy requires C @ X = D. The original problem is converted
    # to the equivalent problem A.T @ X.T = B.T.
    a_left = np.moveaxis(scipy.linalg.sqrtm(eye_inner - product_herm @ product), -1, -2)
    b_left = np.moveaxis(-complex_weight_left, -1, -2)
    complex_weight_left_add = np.moveaxis(np.linalg.solve(a_left, b_left), -1, -2)

    # Right uncertainty weight
    a_right = scipy.linalg.sqrtm(eye_outer - product @ product_herm)
    b_right = complex_weight_right @ complex_nominal
    complex_weight_right_add = np.linalg.solve(a_right, b_right)

    return complex_nominal_add, complex_weight_left_add, complex_weight_right_add


def _compute_minimal_weights_additive(
    complex_weight_left: np.ndarray,
    complex_weight_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute minimal dimension additive uncertainty weight frequency response.

    Parameters
    ----------
    complex_weight_left : np.ndarray
        Left uncertainty weight complex response.
    complex_weight_right : np.ndarray
        Right uncertainty weight complex response.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Minimal left and right uncertainty weight frequency response.

    Raises
    ------
    ValueError
        The number of perturbation outputs is less than the number of system outputs.
    ValueError
        The number of perturbation inputs is less than the number of system inputs.
    """

    # Number of inputs/outputs of system and perturbation (Delta block)
    nbr_outputs_sys = complex_weight_left.shape[1]
    nbr_inputs_sys = complex_weight_right.shape[2]
    nbr_outputs_delta = complex_weight_left.shape[2]
    nbr_inputs_delta = complex_weight_right.shape[1]

    # Compute minimial left uncertainty weight
    if nbr_outputs_sys > nbr_outputs_delta:
        raise ValueError(
            "The number of perturbation outputs (number of left uncertainty weight "
            "columns) must be greater than the number of system outputs (number of "
            "left uncertainty weight rows) for additive uncertainty (got "
            f"{nbr_outputs_delta} perturbations outputs and {nbr_outputs_sys} system "
            "outputs)."
        )
    elif nbr_outputs_sys < nbr_outputs_delta:
        u_left, s_left, _ = np.linalg.svd(complex_weight_left, full_matrices=False)
        complex_weight_left_min = s_left[:, None, :] * u_left  # Scale columns of U
    else:
        complex_weight_left_min = complex_weight_left

    # Compute minimal right uncertainty weight
    if nbr_inputs_sys > nbr_inputs_delta:
        raise ValueError(
            "The number of perturbation inputs (number of right uncertainty weight "
            "rows) must be greater than the number of system inputs (number of "
            "right uncertainty weight columns) for additive uncertainty (got "
            f"{nbr_inputs_delta} perturbations inputs and {nbr_inputs_sys} system "
            "inputs)."
        )
    elif nbr_inputs_sys < nbr_inputs_delta:
        _, s_right, vh_right = np.linalg.svd(complex_weight_right, full_matrices=False)
        complex_weight_right_min = s_right[:, :, None] * vh_right  # Scale rows of V^H
    else:
        complex_weight_right_min = complex_weight_right

    return complex_weight_left_min, complex_weight_right_min


def _convert_frequency_response_data_to_array(
    frequency_response: Union[np.typing.ArrayLike, control.FrequencyResponseData],
) -> np.ndarray:
    """Convert frequency response data into the expected form.

    Parameters
    ----------
    frequency_response : Union[np.typing.ArrayLike, control.FrequencyResponseData],
        Frequency response data in arbitrary form.

    Returns
    -------
    np.ndarray
        Frequency response data in expected array form.

    Raises
    ------
    ValueError
        The dimensions of the `frequency_response` input are incompatible with the
        expected form.
    """
    if isinstance(frequency_response, control.FrequencyResponseData):
        complex_response = np.array(frequency_response.complex, dtype=complex)

        # SISO `FrequencyResponseData` objects return 1D arrays for frequency response
        # data and 3D arrays for MIMO systems. `dkpy` uses a 3D array in all cases, so
        # the 1D array is converted into a 3D array.
        if complex_response.ndim == 1:
            complex_response = complex_response[None, None, :]

        # The `control.FrequencyResponseData.complex` uses an array with shape
        # (output, input, frequency) whereas the format used in `dkpy` is
        # (frequency, output, input). Therefore, a transpose to switch the axes is
        # required.
        complex_response = complex_response.transpose(2, 0, 1)
    else:
        complex_response = np.array(frequency_response, dtype=complex)
        if complex_response.ndim != 3:
            raise ValueError(
                f"The dimension of `complex_response` is {complex_response.ndim}, "
                "whereas it should be 3 (frequency, output, input)."
            )
    return complex_response


def _convert_frequency_response_list_to_array(
    frequency_response_list: Union[np.typing.ArrayLike, control.FrequencyResponseList],
) -> np.ndarray:
    """Convert list of frequency response data into the expected form.

    Parameters
    ----------
    frequency_response_list : Union[np.typing.ArrayLike, control.FrequencyResponseData],
        Frequency response data in arbitrary form.

    Returns
    -------
    np.ndarray
        Frequency response data list in expected array form.

    Raises
    ------
    ValueError
        The dimensions of the `frequency_response_list` input are incompatible with the
        expected form.
    """
    if isinstance(frequency_response_list, control.FrequencyResponseList):
        complex_response_list = []
        for frequency_response in frequency_response_list:
            complex_response = np.array(frequency_response.complex, dtype=complex)

            # SISO `FrequencyResponseData` objects return 1D arrays for frequency
            # response data and 3D arrays for MIMO systems. `dkpy` uses a 3D array in
            # all cases, so the 1D array is converted into a 3D array.
            if complex_response.ndim == 1:
                complex_response = complex_response[None, None, :]

            # The `control.FrequencyResponseData.complex` uses an array with shape
            # (output, input, frequency) whereas the format used in `dkpy` is
            # (frequency, output, input). Therefore, a transpose to switch the axes is
            # required.
            complex_response = complex_response.transpose(2, 0, 1)
            complex_response_list.append(complex_response)
        complex_response_list = np.array(complex_response_list, dtype=complex)
    else:
        complex_response_list = np.array(frequency_response_list, dtype=complex)
        if complex_response_list.ndim != 4:
            raise ValueError(
                "The dimension of `complex_response_list` is "
                f"{complex_response_list.ndim}, whereas it should be 4 (off-nominals, "
                "frequency, output, input)."
            )
    return complex_response_list


def plot_magnitude_response_uncertain_model_set(
    complex_response_nom: Union[np.ndarray, control.FrequencyResponseData],
    complex_response_offnom_list: Union[np.ndarray, control.FrequencyResponseList],
    omega: np.ndarray,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_nom_kw: Dict[str, Any] = {},
    plot_offnom_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Union[Axes, np.ndarray], Legend]:
    """Plot magnitude response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : Union[np.ndarray, control.FrequencyResponseList]
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_nom_kw : Dict[str, Any]
        Keyword arguments for the nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    plot_offnom_kw : Dict[str, Any]
        Keyword arguments for the off-nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Convert frequency response data to expected type
    complex_response_nom = _convert_frequency_response_data_to_array(
        complex_response_nom
    )
    complex_response_offnom_list = _convert_frequency_response_list_to_array(
        complex_response_offnom_list
    )

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]
    num_outputs = complex_response_offnom_list.shape[2]
    num_inputs = complex_response_offnom_list.shape[3]

    # Nominal model plot keyword arguments
    plot_nom_kwargs = {"color": "C0", "label": "Nominal"}
    plot_nom_kwargs.update(plot_nom_kw)

    # Off-nominal model plot keyword arguments
    plot_offnom_kwargs = {"color": "C1", "alpha": 0.25, "label": "Off-Nominal"}
    plot_offnom_kwargs.update(plot_offnom_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"sharex": True, "layout": "constrained"}
    subplot_kwargs.update(subplot_kw)

    # Initialize figure
    fig, ax = plt.subplots(num_outputs, num_inputs, squeeze=False, **subplot_kwargs)

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        magnitude_offnom = np.abs(complex_response_offnom_list[idx_offnom, :, :, :])
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].plot(
                    omega / (2 * np.pi) if hz else omega,
                    control.mag2db(magnitude_offnom[:, idx_output, idx_input])
                    if db
                    else magnitude_offnom[:, idx_output, idx_input],
                    **plot_offnom_kwargs,
                )

    # Nominal frequency response
    magnitude_nom = np.abs(complex_response_nom)
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].plot(
                omega / (2 * np.pi) if hz else omega,
                control.mag2db(magnitude_nom[:, idx_output, idx_input])
                if db
                else magnitude_nom[:, idx_output, idx_input],
                **plot_nom_kwargs,
            )

    # Plot settings
    for ax_output in ax:
        for ax_output_input in ax_output:
            ax_output_input.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
            ax_output_input.grid()
            if frequency_log_scale:
                ax_output_input.set_xscale("log")
    for idx_input in range(num_inputs):
        ax[-1, idx_input].set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    legend = fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )

    return fig, ax, legend


def plot_phase_response_uncertain_model_set(
    complex_response_nom: Union[np.ndarray, control.FrequencyResponseData],
    complex_response_offnom_list: Union[np.ndarray, control.FrequencyResponseList],
    omega: np.ndarray,
    deg: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_nom_kw: Dict[str, Any] = {},
    plot_offnom_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Union[Axes, np.ndarray], Legend]:
    """Plot phase response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : Union[np.ndarray, control.FrequencyResponseList]
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_nom_kw : Dict[str, Any]
        Keyword arguments for the nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    plot_offnom_kw : Dict[str, Any]
        Keyword arguments for the off-nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Convert frequency response data to expected type
    complex_response_nom = _convert_frequency_response_data_to_array(
        complex_response_nom
    )
    complex_response_offnom_list = _convert_frequency_response_list_to_array(
        complex_response_offnom_list
    )

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]
    num_outputs = complex_response_offnom_list.shape[2]
    num_inputs = complex_response_offnom_list.shape[3]

    # Nominal model plot keyword arguments
    plot_nom_kwargs = {"color": "C0", "label": "Nominal"}
    plot_nom_kwargs.update(plot_nom_kw)

    # Off-nominal model plot keyword arguments
    plot_offnom_kwargs = {"color": "C1", "alpha": 0.25, "label": "Off-Nominal"}
    plot_offnom_kwargs.update(plot_offnom_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"sharex": True, "layout": "constrained"}
    subplot_kwargs.update(subplot_kw)

    # Initialize figure
    fig, ax = plt.subplots(num_outputs, num_inputs, squeeze=False, **subplot_kwargs)

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        phase_offnom = np.angle(complex_response_offnom_list[idx_offnom, :, :, :])
        for idx_input in range(num_inputs):
            for idx_output in range(num_outputs):
                ax[idx_output, idx_input].plot(
                    omega / (2 * np.pi) if hz else omega,
                    180 / np.pi * phase_offnom[:, idx_output, idx_input]
                    if deg
                    else phase_offnom[:, idx_output, idx_input],
                    **plot_offnom_kwargs,
                )

    # Nominal frequency response
    phase_nom = np.angle(complex_response_nom)
    for idx_input in range(num_inputs):
        for idx_output in range(num_outputs):
            ax[idx_output, idx_input].plot(
                omega / (2 * np.pi) if hz else omega,
                180 / np.pi * phase_nom[:, idx_output, idx_input]
                if deg
                else phase_nom[:, idx_output, idx_input],
                **plot_nom_kwargs,
            )

    # Plot settings
    for ax_output in ax:
        for ax_output_input in ax_output:
            ax_output_input.set_ylabel("Phase (deg)" if deg else "Phase (rad)")
            ax_output_input.grid()
            if frequency_log_scale:
                ax_output_input.set_xscale("log")
    for idx_input in range(num_inputs):
        ax[-1, idx_input].set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    legend = fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )

    return fig, ax, legend


def plot_singular_value_response_uncertain_model_set(
    complex_response_nom: Union[np.ndarray, control.FrequencyResponseData],
    complex_response_offnom_list: Union[np.ndarray, control.FrequencyResponseList],
    omega: np.ndarray,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_nom_kw: Dict[str, Any] = {},
    plot_offnom_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Axes, Legend]:
    """Plot singular value response of nominal model and set of off-nominal models.

    Parameters
    ----------
    complex_response_nom : Union[np.ndarray, control.FrequencyResponseData]
        Frequency response matrices over a grid of frequencies of the nominal system.
    complex_response_offnom_list : Union[np.ndarray, control.FrequencyResponseList]
        Frequency response matrices over a grid of frequencies of the set of off-nominal
        systems.
    omega : np.ndarray
        Angular frequency grid.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_nom_kw : Dict[str, Any]
        Keyword arguments for the nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    plot_offnom_kw : Dict[str, Any]
        Keyword arguments for the off-nominal model plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Convert frequency response data to expected type
    complex_response_nom = _convert_frequency_response_data_to_array(
        complex_response_nom
    )
    complex_response_offnom_list = _convert_frequency_response_list_to_array(
        complex_response_offnom_list
    )

    # System paramters
    num_offnom = complex_response_offnom_list.shape[0]

    # Nominal model plot keyword arguments
    plot_nom_kwargs = {"color": "C0", "label": "Nominal"}
    plot_nom_kwargs.update(plot_nom_kw)

    # Off-nominal model plot keyword arguments
    plot_offnom_kwargs = {"color": "C1", "alpha": 0.25, "label": "Off-Nominal"}
    plot_offnom_kwargs.update(plot_offnom_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"ncols": 1, "nrows": 1, "layout": "constrained"}
    subplot_kwargs.update(subplot_kw)

    # Initialize figure
    fig, ax = plt.subplots(**subplot_kwargs)

    # Off-nominal frequency response
    for idx_offnom in range(num_offnom):
        sval_offnom = np.linalg.svdvals(
            complex_response_offnom_list[idx_offnom, :, :, :]
        )
        for idx_sval in range(sval_offnom.shape[1]):
            ax.semilogx(
                omega / (2 * np.pi) if hz else omega,
                control.mag2db(sval_offnom[:, idx_sval])
                if db
                else sval_offnom[:, idx_sval],
                **plot_offnom_kwargs,
            )

    # Nominal frequency response
    sval_nom = np.linalg.svdvals(complex_response_nom)
    for idx_sval in range(sval_nom.shape[1]):
        ax.plot(
            omega / (2 * np.pi) if hz else omega,
            control.mag2db(sval_nom[:, idx_sval]) if db else sval_nom[:, idx_sval],
            **plot_nom_kwargs,
        )

    # Plot settings
    if frequency_log_scale:
        ax.set_xscale("log")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
    ax.grid()
    ax.set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    legend = fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )

    return fig, ax, legend


def plot_singular_value_response_residual(
    complex_response_residual_dict: Dict[UncertaintyModelId, np.ndarray],
    omega: np.ndarray,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_sval_max_kw: Dict[str, Any] = {},
    plot_sval_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Dict[str, Tuple[Figure, Axes, Legend]]:
    """Plot the singular value response of the uncertainty residuals for different
    unstructured uncertainty models on separate figures.

    Parameters
    ----------
    complex_response_residual_dict : Dict[str, np.ndarray]
        Dictionary of the uncertainty residual frequency response matrices over a grid
        of frequencies for different uncertainty models.
    omega : np.narray
        Angular frequency grid.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_sval_max_kw : Dict[str, Any]
        Keyword arguments for the maximum singular value plot. See [#plot_kw]_ for more
        information on plotting keywords.
    plot_sval_kw : Dict[str, Any]
        Keyword arguments for the singular value plots. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Residual singular value plot keyword arguments
    plot_sval_max_kwargs = {"color": "black"}
    plot_sval_max_kwargs.update(plot_sval_max_kw)

    # Residual maximum singular value plot keyword arguments
    plot_sval_kwargs = {"color": "grey", "alpha": 0.25}
    plot_sval_kwargs.update(plot_sval_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"ncols": 1, "nrows": 1, "layout": "constrained"}
    subplot_kwargs.update(subplot_kw)

    # Initialize dictionary for storing figure and axes
    figure_dict = {}

    # Uncertainty model residual suffixes
    residual_suffix_dict = {
        "additive": "A",
        "multiplicative_input": "I",
        "multiplicative_output": "O",
        "inverse_additive": "iA",
        "inverse_multiplicative_input": "iI",
        "inverse_multiplicative_output": "iO",
    }

    # Iterate over each uncertainty model
    for (
        uncertainty_model_id,
        complex_response_residual,
    ) in complex_response_residual_dict.items():
        # Off-nominal frequency response parameters
        num_offnom = complex_response_residual.shape[0]

        # Uncertainty model suffix
        residual_suffix = residual_suffix_dict[uncertainty_model_id]

        # Compute the singular value and maximum singlar value response of the residuals
        sval_response_residual = np.linalg.svdvals(complex_response_residual)
        sval_max_response_residual = np.max(sval_response_residual, axis=(0, 2))

        # Intialize the plot
        fig, ax = plt.subplots(**subplot_kwargs)

        # Singular value response of the residuals
        for idx_offnom in range(num_offnom):
            for idx_sval in range(sval_response_residual.shape[2]):
                ax.plot(
                    omega / (2 * np.pi) if hz else omega,
                    control.mag2db(sval_response_residual[idx_offnom, :, idx_sval])
                    if db
                    else sval_response_residual[idx_offnom, :, idx_sval],
                    label=rf"$\sigma(E_{{{residual_suffix}}})$",
                    **plot_sval_kwargs,
                )
        # Maximum singular value response of the residuals
        ax.plot(
            omega / (2 * np.pi) if hz else omega,
            control.mag2db(sval_max_response_residual)
            if db
            else sval_max_response_residual,
            label=rf"$\max \; \sigma(E_{{{residual_suffix}}})$",
            **plot_sval_max_kwargs,
        )

        # Plot settings
        if frequency_log_scale:
            ax.set_xscale("log")
        ax.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
        ax.grid()
        ax.set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
        handles, labels = ax.get_legend_handles_labels()
        legend = legend_dict = dict(zip(labels, handles))
        fig.legend(
            labels=legend_dict.keys(),
            handles=legend_dict.values(),
            loc="outside lower center",
            ncol=2,
        )

        figure_dict[uncertainty_model_id] = (fig, ax, legend)

    return figure_dict


def plot_singular_value_response_residual_comparison(
    complex_response_residual_dict: Dict[UncertaintyModelId, np.ndarray],
    omega: np.ndarray,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_sval_max_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Axes, Legend]:
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
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_sval_max_kw : Dict[str, Any]
        Keyword arguments for the maximum singular value plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Residual singular value plot keyword arguments
    plot_sval_max_kwargs = {}
    plot_sval_max_kwargs.update(plot_sval_max_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"ncols": 1, "nrows": 1, "layout": "constrained"}
    subplot_kwargs.update(subplot_kw)

    # Uncertainty model suffixes
    residual_suffix_dict = {
        "additive": "A",
        "multiplicative_input": "I",
        "multiplicative_output": "O",
        "inverse_additive": "iA",
        "inverse_multiplicative_input": "iI",
        "inverse_multiplicative_output": "iO",
    }

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
    fig, ax = plt.subplots(**subplot_kwargs)

    # Maximum singular value reponse of uncertainty residuals
    for (
        uncertainty_model_id,
        sval_max_response_residual,
    ) in sval_max_response_residual_dict.items():
        residual_suffix = residual_suffix_dict[uncertainty_model_id]
        ax.semilogx(
            omega / (2 * np.pi) if hz else omega,
            control.mag2db(sval_max_response_residual)
            if db
            else sval_max_response_residual,
            label=rf"$\max \; {{\sigma}}(E_{{{residual_suffix}}})$",
            **plot_sval_max_kwargs,
        )

    # Plot settings
    if frequency_log_scale:
        ax.set_xscale("log")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
    ax.grid()
    ax.set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    legend = fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=3,
    )

    return fig, ax, legend


def plot_singular_value_response_uncertainty_weight(
    complex_weight: np.ndarray,
    omega: np.ndarray,
    weight_fit: Optional[control.LTI] = None,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_response_kw: Dict[str, Any] = {},
    plot_response_fit_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Union[Axes, np.ndarray], Legend]:
    """Plot the singular value respone of an uncertainty weight.

    Parameters
    ----------
    complex_weight : Dict[str, np.ndarray]
        Dictionary of the uncertainty residual frequency response matrices over a grid
        of frequencies for different uncertainty models.
    omega : np.narray
        Angular frequency grid.
    weight_fit : Optional[control.LTI]
        Fitted uncertainty weight model.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_sval_max_kw : Dict[str, Any]
        Keyword arguments for the maximum singular value plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    """

    # Uncertainty weight response plot keyword arguments
    plot_response_kwargs = {
        "color": "C0",
        "marker": "*",
        "linestyle": " ",
        "label": "Response",
    }
    plot_response_kwargs.update(plot_response_kw)

    # Uncertainty weight fit response plot keyword arguments
    plot_response_fit_kwargs = {
        "color": "C1",
        "linestyle": "-",
        "label": "Fit",
    }
    plot_response_fit_kwargs.update(plot_response_fit_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"sharex": True, "layout": "constrained"}
    subplot_kw = {} if subplot_kw is None else subplot_kw
    subplot_kwargs.update(subplot_kw)

    # Singular value weight response
    sval_weight = np.linalg.svdvals(complex_weight)
    sval_weight = control.mag2db(sval_weight) if db else sval_weight

    # Singular value weight fit response
    if weight_fit is None:
        sval_weight_fit = None
    else:
        frd_weight_fit = control.FrequencyResponseData(weight_fit, omega, squeeze=False)
        response_weight_fit = frd_weight_fit.complex.transpose(2, 0, 1)
        sval_weight_fit = np.linalg.svdvals(response_weight_fit)
    sval_weight_fit = control.mag2db(sval_weight_fit) if db else sval_weight_fit

    # Initialize figure
    fig, ax = plt.subplots(**subplot_kwargs)

    # Plot weight singular value response
    for idx_sval in range(sval_weight.shape[-1]):
        ax.plot(
            omega / (2 * np.pi) if hz else omega,
            sval_weight[:, idx_sval],
            **plot_response_kwargs,
        )
        if sval_weight_fit is not None:
            ax.plot(
                omega / (2 * np.pi) if hz else omega,
                sval_weight_fit[:, idx_sval],
                **plot_response_fit_kwargs,
            )

    # Plot settings
    ax.set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
    ax.grid()
    if frequency_log_scale:
        ax.set_xscale("log")
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    legend = fig.legend(
        labels=legend_dict.keys(),
        handles=legend_dict.values(),
        loc="outside lower center",
        ncol=2,
    )

    return fig, ax, legend


def plot_uncertainty_measure(
    measure: np.ndarray,
    omega: np.ndarray,
    db: bool = True,
    hz: bool = False,
    frequency_log_scale: bool = True,
    plot_kw: Dict[str, Any] = {},
    subplot_kw: Dict[str, Any] = {},
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """Plot uncertainty measure frequency response.

    Parameters
    ----------
    measure : np.ndarray
        Uncertainty measure frequency response.
    omega : np.narray
        Angular frequency grid.
    weight_fit : Optional[control.LTI]
        Fitted uncertainty weight model.
    db : bool
        If True, plot the magnitude in units of dB. Otherwise, plot the magnitude in
        absolute units.
    hz : bool
        If True, plot the frequency in units of Hz. Otherwise, plot the frequency in
        units of rad/s.
    frequency_log_scale : bool
        If True, plot the frequency using a logarithmic axis. Otherwise, plot the
        the frequency using a linear axis.
    plot_sval_max_kw : Dict[str, Any]
        Keyword arguments for the maximum singular value plot. See [#plot_kw]_ for more
        information on plotting keywords.
    subplot_kw : Dict[str, Any]
        Keyword arguments for the subplot. See [#subplot_kw]_ for more information on
        the subplot keywords.

    Returns
    -------
    Tuple[Figure, Union[Axes, np.ndarray], Legend]
        Matplotlib Figure object, Axes object (or np.ndarray of Axes objects), and
        Legend object.

    References
    ----------
    .. [#plot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#subplot_kw] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

    """
    # Plot keyword arguments
    plot_kwargs = {
        "color": "C0",
        "marker": "",
        "linestyle": "-",
    }
    plot_kwargs.update(plot_kw)

    # Subplot keyword arguments
    subplot_kwargs = {"sharex": True, "layout": "constrained"}
    subplot_kw = {} if subplot_kw is None else subplot_kw
    subplot_kwargs.update(subplot_kw)

    # Measure response
    measure = control.mag2db(measure) if db else measure

    # Initialize figure
    fig, ax = plt.subplots(**subplot_kwargs)

    # Plot weight singular value response
    ax.plot(
        omega / (2 * np.pi) if hz else omega,
        measure,
        **plot_kwargs,
    )

    # Plot settings
    ax.set_xlabel("$f$ (Hz)" if hz else r"$\omega$ (rad/s)")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude (-)")
    ax.grid()
    if frequency_log_scale:
        ax.set_xscale("log")

    return fig, ax
