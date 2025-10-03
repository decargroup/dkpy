import numpy as np
import control
import cvxpy
import dkpy
import scipy
from typing import Union, Tuple, Dict, Any, Optional

import slycot
from matplotlib import pyplot as plt


def main():
    # Frequency range
    omega_min = 0.1
    omega_max = 10
    num_omega = 100
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_omega)

    # Ground-truth system
    sys = control.TransferFunction([1, 2, 2], [1, 2.5, 1.5]) * control.TransferFunction(
        1, [1, 0.1]
    )
    sys = sys * control.TransferFunction([1, 3.75, 3.5], [1, 2.5, 13])

    # System frequency response
    frd_sys = control.frequency_response(sys, omega)
    fresp_sys = frd_sys.complex
    mag_sys = np.array(frd_sys.magnitude)
    mag_sys = mag_sys * (1 + np.random.normal(0.0, 0.03, mag_sys.size))

    order = 4
    mosek_params = {
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_INFEAS": 1e-12,
    }
    linear_solver_param = {
        "solver": "MOSEK",
        "verbose": False,
        "mosek_params": mosek_params,
        "warm_start": True,
    }

    weight_log_cheby = fit_magnitude_log_chebyshev_siso(
        omega, mag_sys, order, linear_solver_param=linear_solver_param
    )

    n, A, B, C, D = slycot.sb10yd(
        discfl=0,  # Continuous-time
        flag=1,  # Constrain stable, minimum phase
        lendat=omega.shape[0],
        rfrdat=np.real(fresp_sys),
        ifrdat=np.imag(fresp_sys),
        omega=omega,
        n=order,
        tol=0,  # Length of cache array
    )
    weight_slycot = control.StateSpace(A, B, C, D, dt=0)

    # Estimated Frequency response
    omega_fit = np.logspace(np.log10(omega_min), np.log10(omega_max), 100)
    frd_fit = weight_log_cheby.frequency_response(omega_fit)
    mag_fit = frd_fit.magnitude
    frd_fit_slycot = weight_slycot.frequency_response(omega_fit)
    mag_fit_slycot = frd_fit_slycot.magnitude

    # Plot: Preview log-magnitude response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, control.mag2db(mag_sys), "--*", label="Magnitude Data")
    ax.semilogx(
        omega_fit, control.mag2db(mag_fit), label=f"Fit Magnitude (Order {order})"
    )
    ax.semilogx(
        omega_fit,
        control.mag2db(mag_fit_slycot),
        label=f"Slycot Fit Magnitude (Order {order})",
    )
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    # Plot: Preview magnitude response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, mag_sys, "--*", label="Magnitude Data")
    ax.semilogx(omega_fit, mag_fit, label=f"Fit Magnitude (Order {order})")
    ax.set_ylabel("Magnitude (-)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    # Plot: Preview residual response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, mag_sys**2 / mag_fit**2 - 1, "--*", label="Magnitude Data")
    ax.semilogx(
        omega, mag_sys**2 / mag_fit_slycot**2 - 1, "--*", label=" Slycot Magnitude Data"
    )
    ax.set_ylabel("Residual (-)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    return 0


# TODO: Update docstring
def fit_magnitude_log_chebyshev_siso(
    omega: np.ndarray,
    magnitude_fit: np.ndarray,
    order: int,
    magnitude_upper_bound: Optional[np.ndarray] = None,
    magnitude_lower_bound: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    linear_solver_param: Dict[str, Any] = {},
    tol_bisection: float = 1e-3,
    max_iter_bisection: int = 500,
    num_spec_constr: int = 500,
) -> control.StateSpace:
    # Discrete-time frequency
    theta, alpha = _convert_freq_halfplane_to_disk(omega)

    # Parse frequency-dependent weight
    if weight is None:
        weight = np.ones_like(omega)

    # Discrete-time autocorrelation
    num_auto, den_auto = _fit_autocorrelation_overbound(
        theta,
        magnitude_fit,
        order,
        magnitude_upper_bound,
        magnitude_lower_bound,
        weight,
        linear_solver_param,
        tol_bisection,
        max_iter_bisection,
        num_spec_constr,
    )

    # Discrete-time transfer function
    tf_dt = _compute_spectral_factorization(num_auto, den_auto)

    # Continous-time state-space
    ss_ct = _convert_continuous_to_discrete_bilinear(tf_dt, alpha)

    return ss_ct


def _convert_freq_halfplane_to_disk(omega: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert continous-time angular frequency data on the jw-axis in the s-domain into
    discrete-time frequency data on the unit-circle of the z-domain via a bilinear
    transformation.

    Parameters
    ----------
    omega : np.ndarray
        Continuous-time angular frequency data [rad/s].

    Returns
    -------
    Tuple[np.ndarray, float]
        - Discrete-time frequency data,
        - Equivalent sampling period [s].
    """

    # Angular frequency bounds
    omega_min = omega[0]
    omega_max = omega[-1]

    # Bilinear transformation constant
    alpha = np.sqrt(omega_min * omega_max)

    # Discrete-time frequency
    theta = np.angle((-1j * omega - alpha) / (1j * omega - alpha))

    return theta, alpha


# TODO: Update docstring
def _fit_autocorrelation_overbound(
    theta: np.ndarray,
    magnitude_fit: np.ndarray,
    order: int,
    magnitude_upper_bound: Optional[np.ndarray],
    magnitude_lower_bound: Optional[np.ndarray],
    weight: np.ndarray,
    linear_solver_param: Dict[str, Any],
    tol_bisection: float,
    max_iter_bisection: int,
    num_spec_constr: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a discrete-time biproper autocorrelation overbounding transfer function to
    magnitude data.

    Parameters
    ----------
    theta : np.ndarray
        Discrete-time frequency.
    magnitude : np.ndarray
        Magnitude data.
    order : int
        Biproper transfer function order.
    linear_solver_param : Dict[str, Any]
        Solver parameters for linear feasibility problems.
    tol_bisection : float
        Bisection algorithm numerical tolerance.
    max_iter_bisection : int
        Bisection algorithm maximum number of iterations.
    num_spec_constr : int
        Number of discretization points for spectral factorization constraint
        approximation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Discrete-time autocorrelation transfer function denominator polynomial
          coefficients.
        - Discrete-time autocorrelation transfer function numerator polynomial
          coefficients.
    """

    # Optimization problem parameters
    psi_num = np.array(
        [[np.cos(i * theta_idx) for i in range(order, -1, -1)] for theta_idx in theta]
    )
    psi_den = np.array(
        [[np.cos(i * theta_idx) for i in range(order, 0, -1)] for theta_idx in theta]
    )
    magnitude_fit_sq_inv = np.diag(1 / magnitude_fit**2)
    num_theta = theta.size

    # Spectral factorization constraint
    A_spec, B_spec = _compute_spec_fac_constr_mat(num_spec_constr, order)

    # Lower bound constraint
    if magnitude_lower_bound is not None:
        magnitude_lower_bound_sq_inv = np.diag(1 / magnitude_lower_bound**2)
        A_lower_bound = np.block([[-magnitude_lower_bound_sq_inv @ psi_num, psi_den]])
        B_lower_bound = -np.ones((num_theta, 1))

    # Upper bound constraint
    if magnitude_upper_bound is not None:
        magnitude_upper_bound_sq_inv = np.diag(1 / magnitude_upper_bound**2)
        A_upper_bound = np.block([[magnitude_upper_bound_sq_inv @ psi_num, -psi_den]])
        B_upper_bound = np.ones((num_theta, 1))

    # Autocorrelation transfer function coefficient variables
    num_auto_coef = cvxpy.Variable(shape=(order + 1, 1))
    den_auto_coef = cvxpy.Variable(shape=(order, 1))
    tf_auto_coef = cvxpy.bmat([[num_auto_coef], [den_auto_coef]])

    # Error bounds for bisection algorithm
    t_max = np.max(magnitude_fit) ** 2 / np.min(magnitude_fit) ** 2 - 1
    t_upper = t_max
    t_lower = 0

    # Initialize bisection algorithm
    iter_bisection = 0
    feasibility_status = "infeasible"

    t = cvxpy.Parameter()

    # Bisection minimization of error upper bound
    while np.abs(t_upper - t_lower) >= tol_bisection or feasibility_status != "optimal":
        # Increment bisection iteration count
        iter_bisection += 1

        # Stop bisection at maximum iterations
        if iter_bisection >= max_iter_bisection:
            break

        # Bisect error bound
        t.value = 0.5 * (t_upper + t_lower)

        # Error upper bound log-Chebyshev matrices
        gamma = cvxpy.diag(np.ones_like(weight) + t * 1 / weight)
        gamma_inv = cvxpy.diag(1 / (np.ones_like(weight) + t * 1 / weight))

        # Log-Chebyshev constraint
        A_log_cheby_lower = cvxpy.bmat(
            [[-magnitude_fit_sq_inv @ psi_num, gamma_inv @ psi_den]]
        )
        B_log_cheby_lower = -gamma_inv @ np.ones((num_theta, 1))
        A_log_cheby_upper = cvxpy.bmat(
            [[magnitude_fit_sq_inv @ psi_num, -gamma @ psi_den]]
        )
        B_log_cheby_upper = gamma @ np.ones((num_theta, 1))

        # Total constraint matrix
        A_tot = cvxpy.bmat([[A_log_cheby_upper], [A_log_cheby_lower], [A_spec]])
        B_tot = cvxpy.bmat([[B_log_cheby_upper], [B_log_cheby_lower], [B_spec]])
        if magnitude_lower_bound is not None:
            A_tot = cvxpy.bmat([[A_tot], [A_lower_bound]])
            B_tot = cvxpy.bmat([[B_tot], [B_lower_bound]])
        if magnitude_upper_bound is not None:
            A_tot = cvxpy.bmat([[A_tot], [A_upper_bound]])
            B_tot = cvxpy.bmat([[B_tot], [B_upper_bound]])

        # Linear optimization problem
        objective = cvxpy.Minimize(0)
        constraint = [A_tot @ tf_auto_coef <= B_tot]
        problem = cvxpy.Problem(objective, constraint)
        try:
            problem.solve(**linear_solver_param)
            feasibility_status = problem.status

        except cvxpy.SolverError:
            t_lower = t.value

        if feasibility_status == "optimal":
            t_upper = t.value

        else:
            t_lower = t.value

    if num_auto_coef.value is None or den_auto_coef.value is None:
        raise cvxpy.SolverError("The linear feasibility problem is not feasible.")

    # Autocorrelation polynomial coefficients
    num_auto_coef_opt = num_auto_coef.value.reshape(-1)
    den_auto_coef_opt = np.append(den_auto_coef.value.reshape(-1), 1)

    # Autocorrelation numerator and denominator polynominals
    num_auto = np.concatenate(
        (
            0.5 * num_auto_coef_opt[:-1],
            num_auto_coef_opt[[-1]],
            np.flip(0.5 * num_auto_coef_opt[:-1]),
        )
    )
    den_auto = np.concatenate(
        (
            0.5 * den_auto_coef_opt[:-1],
            den_auto_coef_opt[[-1]],
            np.flip(0.5 * den_auto_coef_opt[:-1]),
        )
    )

    return num_auto, den_auto


def _compute_spec_fac_constr_mat(
    num_spec_constr: int, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the matrices for the approximate linear spectral factorization constraint of
    the form Ax <= B.

    Parameters
    ----------
    num_spec_constr : int
        Number of discretization points for spectral factorization constraint
        approximation.
    order : int
        Biproper transfer function order.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Spectral factorization constraint A matrix
        - Spectral factorization constraint B matrix
    """

    # Discrete-time frequency grid approximation
    theta_spec = np.linspace(0, np.pi, num_spec_constr)

    # Numerator and denominator regression matrices
    psi_num_spec = np.array(
        [
            [np.cos(i * theta_idx) for i in range(order, -1, -1)]
            for theta_idx in theta_spec
        ]
    )
    psi_den_spec = np.array(
        [
            [np.cos(i * theta_idx) for i in range(order, 0, -1)]
            for theta_idx in theta_spec
        ]
    )

    # Linear constraint matrices
    A_spec = np.block(
        [
            [-psi_num_spec, np.zeros((num_spec_constr, order))],
            [np.zeros((num_spec_constr, order + 1)), -psi_den_spec],
        ]
    )
    B_spec = np.block(
        [[np.zeros((num_spec_constr, 1))], [np.ones((num_spec_constr, 1))]]
    )

    return A_spec, B_spec


def _compute_spectral_factorization(
    num_auto: np.ndarray, den_auto: np.ndarray
) -> control.TransferFunction:
    # Autocorrelation numerator and denominator polynominal roots
    roots_auto_num = np.roots(num_auto)
    roots_auto_den = np.roots(den_auto)

    # Stable autocorrelation numerator and denominator polynominal roots
    roots_num_stable = roots_auto_num[np.abs(roots_auto_num) < 1]
    roots_den_stable = roots_auto_den[np.abs(roots_auto_den) < 1]

    # Spectral factorization
    spectral_const_num = np.sqrt(num_auto[0] / np.prod(-roots_num_stable))
    spectral_const_den = np.sqrt(den_auto[0] / np.prod(-roots_den_stable))
    num = np.real(
        spectral_const_num
        * np.flip(np.polynomial.polynomial.polyfromroots(roots_num_stable))
    )
    den = np.real(
        spectral_const_den
        * np.flip(np.polynomial.polynomial.polyfromroots(roots_den_stable))
    )

    # Transfer function
    tf = control.TransferFunction(num, den)

    return tf


def _convert_continuous_to_discrete_bilinear(
    sys_tf: control.TransferFunction,
    alpha: float,
) -> control.StateSpace:
    # Convert transfer function to state-space system
    sys_ss = control.StateSpace(control.tf2ss(sys_tf))

    # Discrete-time state-space matrices
    Ad = sys_ss.A
    Bd = sys_ss.B
    Cd = sys_ss.C
    Dd = sys_ss.D

    # Additional matrices
    In = np.eye(Ad.shape[0])
    iden_Ad_inv = np.linalg.solve(In + Ad, In)

    # Continous-time matrices
    Ac = alpha * (Ad - In) @ iden_Ad_inv
    Bc = alpha * (In - (Ad - In) @ iden_Ad_inv) @ Bd
    Cc = Cd @ iden_Ad_inv
    Dc = Dd - Cd @ iden_Ad_inv @ Bd

    # Continuous-time system
    sys_ss = control.StateSpace(Ac, Bc, Cc, Dc)

    return sys_ss


if __name__ == "__main__":
    main()
