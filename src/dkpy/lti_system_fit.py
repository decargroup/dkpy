"""LTI system fit utilities."""

from typing import Any, Dict, Union, Optional, Tuple, List

import numpy as np
import control
import cvxpy

__all__ = [
    "fit_magnitude_siso_ct",
    "fit_power_siso_dt",
]


def fit_magnitude_siso_ct(
    magnitude_fit: np.ndarray,
    omega: np.ndarray,
    order: int,
    magnitude_upper_bound: Optional[np.ndarray] = None,
    magnitude_lower_bound: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
) -> control.StateSpace:
    """Fit a stable and minimum-phase biproper SISO system to magnitude.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequencies (rad/s).
    magnitude_fit : np.ndarray
        Magnitude response to fit the LTI system.
    order : int
        Order of the LTI system.
    magnitude_upper_bound : Optional[np.ndarray]
        Magnitude response for the upper bound constraint on the fitted LTI system
        magnitude response.
    magnitude_lower_bound : Optional[np.ndarray]
        Magnitude response for the lower bound constraint on the fitted LTI system
        magnitude response.
    weight : np.ndarray
        Frequency-dependent weight to encode bandwidths over which to prioritize the
        accuracy of the LTI system fit.
    solver_param : Dict[str, Any]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum allowable number of iterations in the bisection algorithm.
    max_iter_bisection_init : int
        Maximum number of iterations for the bisection algorithm initialization.
    nbr_power_constraint : Optional[np.ndarray]
        Number of frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    control.StateSpace
        Fitted state-space system.

    Notes
    -----
    The algorithm fits the LTI system to the magnitude data as follows:
        1) Pre-warp the continuous-time frequencies to discrete-time frequencies using a
           bilinear transformation.
        2) Fit a discrete-time power spectrum model to the square of the magnitude data.
        3) Extract the stable and minimum-phase spectral factor from the power spectrum
           model.
        4) Convert the discrete-time spectral factor model to a continuous-time model
           using the bilinear transformation used in step 1).

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Compute power spectrum
    power_fit = np.abs(magnitude_fit) ** 2
    power_upper_bound = (
        np.abs(magnitude_upper_bound) ** 2
        if magnitude_upper_bound is not None
        else None
    )
    power_lower_bound = (
        np.abs(magnitude_lower_bound) ** 2
        if magnitude_lower_bound is not None
        else None
    )

    # Pre-warp frequency (continuous- to discrete-time) with bilinear transformation
    alpha = np.sqrt(omega[0] * omega[-1])
    theta = 2 * np.atan(omega / alpha)

    # Parse frequency-dependent weight
    if weight is None:
        weight = np.ones_like(omega)

    # Discrete-time power spectrum fit coefficients
    num_power_coef, den_power_coef = fit_power_siso_dt(
        power_fit,
        theta,
        order,
        weight,
        power_upper_bound,
        power_lower_bound,
        solver_params,
        tol_bisection,
        max_iter_bisection,
        max_iter_bisection_init,
        nbr_power_constraint,
    )

    # Discrete-time spectral factor model
    num_factor_coef, den_factor_coef = _compute_spectral_factor_siso_dt(
        num_power_coef, den_power_coef
    )
    tf_factor_dt = control.TransferFunction(num_factor_coef, den_factor_coef, True)

    # Continous-time state-space
    ss_factor_ct = _convert_discrete_to_continuous_bilinear(tf_factor_dt, alpha)

    return ss_factor_ct


def fit_power_siso_dt(
    power_fit: np.ndarray,
    theta: np.ndarray,
    order: int,
    weight: Optional[np.ndarray] = None,
    power_upper_bound: Optional[np.ndarray] = None,
    power_lower_bound: Optional[np.ndarray] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    tol_bisection: float = 1e-5,
    max_iter_bisection: int = 100,
    max_iter_bisection_init: int = 15,
    nbr_power_constraint: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a biproper discrete-time SISO power spectrum transfer function.

    Parameters
    ----------
    power_fit : np.ndarray
        Power spectrum data to fit model.
    theta : np.ndarray
        Discrete-time angular frequencies (-). The frequencies range from [0, pi].
    order : int
        Order of the power spectrum model.
    weight : Optional[np.ndarray]
        Frequency-dependent weight for fit accuracy.
    power_upper_bound : np.ndarray
        Power spectrum data to upper bound the model frequency response.
    power_lower_bound : np.ndarray
        Power spectrum data to lower bound the model frequency response.
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
    Tuple[np.ndarray, np.ndarray]
        Numerator and denominator polynomial coefficients of the power spectrum.

    Notes
    -----
    The algorithm fits a power spectrum of the form
                b_{m} z^{-m} + ... + b_{0} + ... + b_{m} z^{m}
        R(z) =  ----------------------------------------------
                a_{m} z^{-m} + ... + a_{0} + ... + b_{m} z^{m}

    The algorithm uses a log-Chebyshev approximation method, which results in an
    optimization problem that is solved by bisecting a bound on the fit error and
    solving linear feasibility problems [#log_chebyshev]_.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html

    .. [#log_chebyshev] Shao-Po Wu, S. Boyd and L. Vandenberghe, "FIR filter design via
    semidefinite programming and spectral factorization," Proceedings of 35th IEEE
    Conference on Decision and Control, Kobe, Japan, 1996, pp. 271-276 vol.1,
    doi: 10.1109/CDC.1996.574313.
    """

    # Parse arguments
    solver_params = _parse_solver_param(solver_params)
    weight = np.ones_like(theta) if weight is None else weight

    # Optimization variables and parameters
    num_coef = cvxpy.Variable(shape=order + 1, name="num_coef")
    den_coef = cvxpy.Variable(shape=order + 1, name="den_coef")
    error_bound = cvxpy.Parameter(shape=(), name="error_bound")

    # Constraints
    constraint_list = _generate_constraints_power_siso_dt(
        num_coef,
        den_coef,
        error_bound,
        power_fit,
        power_upper_bound,
        power_lower_bound,
        theta,
        order,
        weight,
        nbr_power_constraint,
    )

    # Optimization problem
    objective = cvxpy.Minimize(1)
    problem = cvxpy.Problem(objective, constraint_list)

    # Bisection solution
    error_bound_max = _initialize_optimization_bisection_upper_bound(
        problem, error_bound, solver_params, max_iter_bisection_init
    )
    error_bound_min = 0
    _solve_optimization_bisection(
        problem,
        error_bound,
        error_bound_max,
        error_bound_min,
        tol_bisection,
        max_iter_bisection,
        solver_params,
    )

    # Extract power spectrum coefficients
    num_coef_fit = np.array(num_coef.value)
    den_coef_fit = np.array(den_coef.value)

    # Construct power spectrum polynomials from unique coefficients
    num_fit = np.concatenate([num_coef_fit[::-1], num_coef_fit[1:]])
    den_fit = np.concatenate([den_coef_fit[::-1], den_coef_fit[1:]])

    return num_fit, den_fit


def _generate_constraints_power_siso_dt(
    num_coef: cvxpy.Variable,
    den_coef: cvxpy.Variable,
    error_bound: cvxpy.Parameter,
    power_fit: np.ndarray,
    power_upper_bound: Optional[np.ndarray],
    power_lower_bound: Optional[np.ndarray],
    theta: np.ndarray,
    order: int,
    weight: np.ndarray,
    nbr_power_constraints: int,
) -> List[cvxpy.Constraint]:
    """Generate the constraints for the discrete-time SISO power spectrums.

    Parameters
    ----------
    num_coef : cvxpy.Variable
        Power spectrum numerator coefficient variable.
    den_coef : cvxpy.Variable
        Power spectrum denominator coefficient variable.
    error_bound : cvxpy.Parameter
        Power spectrum fit error bound that is minimized by the bisection algorithm.
    power_fit: np.ndarray
        Power spectrum data to fit model.
    power_upper_bound : np.ndarray
        Power spectrum data to upper bound the model frequency response.
    power_lower_bound : np.ndarray
        Power spectrum data to lower bound the model frequency response.
    theta : np.ndarray
        Discrete-time angular frequencies (-). The frequencies range from [0, pi].
    order : int
        Order of the power spectrum model.
    weight : np.ndarray
        Frequency-dependent weight for fit accuracy.
    nbr_power_constraint : int
        Discrete-time frequencies to enforce non-negativity of power spectrum.

    Returns
    -------
    List[cvxpy.Constraint]
        List of constraints.
    """
    # Fit error and magnitude bound constraints
    constraint_bound_list = _construct_bound_constraints_siso_dt(
        num_coef,
        den_coef,
        error_bound,
        power_fit,
        power_upper_bound,
        power_lower_bound,
        theta,
        order,
        weight,
    )

    # Spectral factorization constraint
    constraint_nonneg_list = _construct_spectral_factorization_constraint_siso_dt(
        num_coef,
        den_coef,
        order,
        nbr_power_constraints,
    )

    # Normalization of the transfer function coefficients
    constraint_normalization_list = [den_coef[-1] == 1]

    constraint_list = []
    constraint_list += (
        constraint_bound_list + constraint_nonneg_list + constraint_normalization_list
    )

    return constraint_list


def _construct_bound_constraints_siso_dt(
    num_coef: cvxpy.Variable,
    den_coef: cvxpy.Variable,
    error_bound: cvxpy.Parameter,
    power_fit: np.ndarray,
    power_upper_bound: Optional[np.ndarray],
    power_lower_bound: Optional[np.ndarray],
    theta: np.ndarray,
    order: int,
    weight: np.ndarray,
) -> List[cvxpy.Constraint]:
    """Construct the fit error and power spectrum bound linear constraints.

    The fit error and power spectrum bound linear constraints are used in the fit of
    discrete-time SISO power spectrums.

    Parameters
    ----------
    num_coef : cvxpy.Variable
        Power spectrum numerator coefficient variable.
    den_coef : cvxpy.Variable
        Power spectrum denominator coefficient variable.
    error_bound : cvxpy.Parameter
        Fit error parameter that is minimized by the bisection algorithm.
    power_fit: np.ndarray
        Power spectrum data to fit model.
    power_upper_bound : np.ndarray
        Power spectrum data to upper bound the model frequency response.
    power_lower_bound : np.ndarray
        Power spectrum data to lower bound the model frequency response.
    theta : np.ndarray
        Discrete-time angular frequencies (-). The frequencies range from [0, pi].
    order : int
        Order of the power spectrum model.
    weight : np.ndarray
        Frequency-dependent weight for fit accuracy.

    Returns
    -------
    List[cvxpy.Constraint]
        List of fit error and magnitude bound linear constraints.
    """

    # Initialize constraint lists
    constraint_fit_upper_list = []
    constraint_fit_lower_list = []
    constraint_bound_upper_list = []
    constraint_bound_lower_list = []

    # Check if the upper/lower bound magnitude is the same as the fit magnitude. If it
    # is, the upper/lower fit error constraint may be removed as it is made redundant by
    # the bound constraint.
    is_power_fit_upper_bound = (
        np.allclose(power_fit, power_upper_bound)
        if power_upper_bound is not None
        else False
    )
    is_power_fit_lower_bound = (
        np.allclose(power_fit, power_lower_bound)
        if power_lower_bound is not None
        else False
    )

    # Construct constraints at each frequency
    for idx in range(theta.size):
        # Constraint data
        theta_idx = theta[idx]
        power_idx = power_fit[idx]
        weight_idx = weight[idx]

        # Constraint data matrices
        cosine_row = np.array(
            [2 * np.cos(idx_coef * theta_idx) for idx_coef in range(order + 1)]
        )
        cosine_row[0] = 1

        # Fit error upper bound constraint
        if not is_power_fit_upper_bound:
            constraint_fit_upper = (
                -(1 + error_bound / weight_idx)
                * power_idx
                * cvxpy.vdot(cosine_row, den_coef)
                + cvxpy.vdot(cosine_row, num_coef)
                <= 0
            )
            constraint_fit_upper_list.append(constraint_fit_upper)

        # Fit error lower bound constraint
        if not is_power_fit_lower_bound:
            constraint_fit_lower = (
                power_idx * cvxpy.vdot(cosine_row, den_coef)
                - (1 + error_bound / weight_idx) * cvxpy.vdot(cosine_row, num_coef)
                <= 0
            )
            constraint_fit_lower_list.append(constraint_fit_lower)

        # Upper bound constraint
        if power_upper_bound is not None:
            power_upper_idx = power_upper_bound[idx]
            constraint_bound_upper = (
                -power_upper_idx * cvxpy.vdot(cosine_row, den_coef)
                - cvxpy.vdot(cosine_row, num_coef)
                <= 0
            )
            constraint_bound_upper_list.append(constraint_bound_upper)

        # Lower bound constraint
        if power_lower_bound is not None:
            power_lower_idx = power_lower_bound[idx]
            constraint_bound_lower = (
                power_lower_idx * cvxpy.vdot(cosine_row, den_coef)
                - cvxpy.vdot(cosine_row, num_coef)
                <= 0
            )
            constraint_bound_lower_list.append(constraint_bound_lower)

    return (
        constraint_fit_upper_list
        + constraint_fit_lower_list
        + constraint_bound_upper_list
        + constraint_bound_lower_list
    )


def _construct_spectral_factorization_constraint_siso_dt(
    num_coef: cvxpy.Variable,
    den_coef: cvxpy.Variable,
    order: int,
    nbr_theta: np.ndarray,
) -> List[cvxpy.Constraint]:
    """Construct power spectrum spectral factorization constraints.

    The power spectrum spectral factorization constraints are used in the fit of
    discrete-time SISO power spectrums.

    Parameters
    ----------
    num_coef : cvxpy.Variable
        Power spectrum numerator coefficient variable.
    den_coef : cvxpy.Variable
        Power spectrum denominator coefficient variable.
    order : int
        Order of the power spectrum model.
    nbr_theta : int
        Number of frequencies to enforce constraint.

    Returns
    -------
    List[cvxpy.Constraint]
        List of spectral factorization (power spectrum non-negativity) linear
        constraints.
    """

    # Frequencies to enforce power spectrum non-negativity
    theta = np.linspace(0, np.pi, nbr_theta)

    constraint_spectral_factorization_list = []
    for theta_idx in theta:
        # Constraint data matrices
        cosine_row = np.array(
            [2 * np.cos(idx_coef * theta_idx) for idx_coef in range(order + 1)]
        )
        cosine_row[0] = 1

        # Spectral factorization constraints
        constraint_num = cvxpy.vdot(cosine_row, num_coef) >= 0
        constraint_den = cvxpy.vdot(cosine_row, den_coef) >= 0
        constraint_spectral_factorization_list += [constraint_num, constraint_den]

    return constraint_spectral_factorization_list


def _initialize_optimization_bisection_upper_bound(
    problem: cvxpy.Problem,
    objective: cvxpy.Parameter,
    solver_params: Dict[str, Any],
    max_iter_bisection_init: int = 15,
) -> float:
    """Initialize the upper bound of a bisection optimization problem.

    Parameters
    ----------
    problem : cvxpy.Problem
        Optimization problem.
    objective : cvxpy.Parameter
        Objective function that the bisection algorithm minimizes.
    solver_params : Dict[str, Any]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.
    max_iter_bisection_init : int
        Maximum allowable number of iterations in the bisection initialization.

    Returns
    -------
    float
        Feasible upper bound for the bisection optimization algorithm.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """
    objective.value = 0.5

    for _ in range(max_iter_bisection_init):
        objective.value *= 2
        try:
            problem.solve(**solver_params)
            feasibility_status = problem.status
        except cvxpy.SolverError:
            break

        if feasibility_status == "optimal":
            return objective.value

    raise ValueError(
        "Unable to determine a upper bound on the objective to initialize "
        "the bisection algorithm."
    )


def _solve_optimization_bisection(
    problem: cvxpy.Problem,
    objective: cvxpy.Parameter,
    objective_max: float,
    objective_min: float,
    tol_bisection: float,
    max_iter_bisection: int,
    solver_params: Dict[str, Any],
) -> None:
    """Solve an optimization problem using a bisection algorithm.

    Parameters
    ----------
    problem : cvxpy.Problem
        Optimization problem.
    objective : cvxpy.Parameter
        Objective function that the bisection algorithm minimizes.
    objective_max : float
        Initial upper bound on objective.
    objective_min : float
        Initial lower bound on objective.
    tol_bisection : float
        Numerical tolerance for the bisection algorithm.
    max_iter_bisection : int
        Maximum allowable number of iterations in the bisection algorithm.
    solver_params : Dict[str, Any]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """

    # Initialize bisection algorithm
    iter_bisection = 0
    feasibility_status = "infeasible"

    # Bisection minimization of objective
    while (
        np.abs(objective_max - objective_min) >= tol_bisection
        or feasibility_status != "optimal"
    ):
        # Increment bisection iteration count
        iter_bisection += 1

        # Stop bisection at maximum iterations
        if iter_bisection >= max_iter_bisection:
            break

        # Bisect objective function
        objective.value = 0.5 * (objective_max + objective_min)
        try:
            problem.solve(**solver_params)
            feasibility_status = problem.status
        except cvxpy.SolverError:
            objective_min = objective.value

        # Update upper and lower bisection bounds
        if feasibility_status == "optimal":
            objective_max = objective.value
        else:
            objective_min = objective.value


def _compute_spectral_factor_siso_dt(
    num_power: np.ndarray, den_power: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the spectral factorization of a discrete-time SISO transfer function.

    Parameters
    ----------
    num_power : np.ndarray
        Power spectrum numerator polynomial.
    den_power : np.ndarray
        Power spectrum denominator polynomial.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spectral factor numerator and denominator polynomial.

    Notes
    -----
    The algorithm takes a discrete-time power spectrum of the form
                b_{m} z^{-m} + ... + b_{0} + ... + b_{m} z^{m}
        R(z) =  ----------------------------------------------
                a_{m} z^{-m} + ... + a_{0} + ... + b_{m} z^{m}

    and computes the spectral factor of the form
                d_{m} z^{m} + ... + d_{0}
        F(z) =  -------------------------
                c_{m} z^{m} + ... + c_{0}

    The algorithm computes the spectral factor by computing the poles and zeros of the
    power spectrum. Then, the DC gain, stable poles, and minimum-phase zeros are
    extracted and used to construct the stable and minimum-phase spectral factor.
    """

    # Power spectrum poles and zeros
    zeros_power = np.roots(num_power)
    poles_power = np.roots(den_power)

    # Stable and minimum-phase power spectrum poles and zeros
    zeros_power_stable = zeros_power[np.abs(zeros_power) < 1]
    poles_power_stable = poles_power[np.abs(poles_power) < 1]

    # Spectral factorization
    spectral_const_num = np.sqrt(num_power[0] / np.prod(-zeros_power_stable))
    spectral_const_den = np.sqrt(den_power[0] / np.prod(-poles_power_stable))
    num_factor = np.real(
        spectral_const_num
        * np.polynomial.polynomial.polyfromroots(zeros_power_stable)[::-1]
    )
    den_factor = np.real(
        spectral_const_den
        * np.polynomial.polynomial.polyfromroots(poles_power_stable)[::-1]
    )

    return num_factor, den_factor


def _convert_discrete_to_continuous_bilinear(
    sys_dt: Union[control.TransferFunction, control.StateSpace],
    alpha: float,
) -> control.StateSpace:
    """Convert a discrete-time system to continuous-time with a bilinear transformation.

    Parameters
    ----------
    sys_dt : Union[control.TransferFunction, control.StateSpace]
        Discrete-time system.
    alpha : float
        Bilinear transformation constant.

    Returns
    -------
    control.StateSpace
        Continuous-time system.
    """
    # Convert transfer function to state-space system
    sys_dt = control.StateSpace(control.ss(sys_dt))

    # Discrete-time state-space matrices
    Ad = sys_dt.A
    Bd = sys_dt.B
    Cd = sys_dt.C
    Dd = sys_dt.D

    # Additional matrices
    In = np.eye(Ad.shape[0])
    In_Ad_inv = np.linalg.solve(In + Ad, In)

    # Continous-time matrices
    Ac = alpha * (Ad - In) @ In_Ad_inv
    Bc = alpha * (In - (Ad - In) @ In_Ad_inv) @ Bd
    Cc = Cd @ In_Ad_inv
    Dc = Dd - Cd @ In_Ad_inv @ Bd

    # Continuous-time system
    sys_dt = control.StateSpace(Ac, Bc, Cc, Dc)

    return sys_dt


def _parse_solver_param(solver_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse solver parameters and ensure warm-start is enabled.

    Parameters
    ----------
    solver_params: Optional[Dict[str, Any]]
        Solver parameters for the optimization problem. These are keyword arguments for
        `cvxpy.Problem.solve()` [#cvxpy_solver]_.

    Returns
    -------
    Dict[str, Any]
        Parsed solver parameters for the optimization problem. These are keyword
        arguments for `cvxpy.Problem.solve()` [#cvxpy_solver]_.

    References
    ----------
    .. [#cxvpy_solver] https://www.cvxpy.org/tutorial/solvers/index.html
    """
    # Default solver parameters
    solver_param_default = {
        "solver": "CLARABEL",
        "warm_start": True,
        "verbose": False,
    }

    if solver_params is None:
        return solver_param_default
    else:
        # Ensure that `warm_start` is set to True as the bisection algorithm is greatly
        # sped up if this holds.
        solver_params["warm_start"] = True
        return solver_params
