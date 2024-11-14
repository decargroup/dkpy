"""Structured singular value computation."""

__all__ = [
    "StructuredSingularValue",
    "SsvLmiBisection",
]

import abc
import warnings
from typing import Any, Dict, Optional, Tuple

import cvxpy
import joblib
import numpy as np
import scipy.linalg

from . import utilities


class StructuredSingularValue(metaclass=abc.ABCMeta):
    """Structured singular value base class."""

    @abc.abstractmethod
    def compute_ssv(
        self,
        N_omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Compute structured singular value.

        Parameters
        ----------
        N_omega : np.ndarray
            Closed-loop transfer function evaluated at each frequency.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
            Structured singular value at each frequency, D-scales at each
            frequency, and solution information. If the structured singular
            value cannot be computed, the first two elements of the tuple are
            ``None``, but solution information is still returned.
        """
        raise NotImplementedError()


class SsvLmiBisection(StructuredSingularValue):
    """Structured singular value using an LMI approach with bisection.

    Synthesis method based on Section 4.25 of [CF24]_.

    Examples
    --------

    Structured singular value computation from [SP06]_

    >>> P, n_y, n_u, K = example_skogestad2006
    >>> block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    >>> omega = np.logspace(-3, 3, 61)
    >>> N = P.lft(K)
    >>> N_omega = N(1j * omega)
    >>> mus, Ds, info = dkpy.SsvLmiBisection(n_jobs=None).compute_ssv(
    ...     N_omega,
    ...     block_structure,
    ... )
    >>> float(np.max(mus))
    5.7726
    """

    def __init__(
        self,
        bisection_atol: float = 1e-5,
        bisection_rtol: float = 1e-4,
        max_iterations: int = 100,
        initial_guess: float = 10,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = -1,
        objective: str = "constant",
    ):
        """Instantiate :class:`SsvLmiBisection`.

        Solution accuracy depends strongly on the selected solver and
        tolerances. Setting the solver and its tolerances in ``solver_params``
        and setting ``lmi_strictness`` manually is recommended, rather than
        relying on the default settings.

        Parameters
        ----------
        bisection_atol : float
            Bisection absolute tolerance.
        bisection_rtol : float
            Bisection relative tolerance.
        max_iterations : int
            Maximum number of bisection iterations.
        initial_guess : float
            Initial guess for bisection.
        lmi_strictness : Optional[float]
            Strictness for linear matrix inequality constraints. Should be
            larger than the solver tolerance. If ``None``, then it is
            automatically set to 10x the solver's largest absolute tolerance.
        solver_params : Optional[Dict[str, Any]]
            Dictionary of keyword arguments for :func:`cvxpy.Problem.solve`.
            Notable keys are ``'solver'`` and ``'verbose'``. Additional keys
            used to set solver tolerances are solver-dependent. A definitive
            list can be found at [#solvers]_.
        n_jobs : Optional[int]
            Number of processes to use to parallelize the bisection. Set to
            ``None`` for a single thread, or set to ``-1`` (default) to use all
            CPUs. See [#jobs]_.
        objective : str
            Set to ``'constant'`` to solve a feasibility problem at each
            bisection iteration. Set to ``'minimize'`` to minimize the trace of
            the slack variable instead, which may result in better numerical
            conditioning.

        References
        ----------
        .. [#solvers] https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        .. [#jobs] https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib-parallel
        """
        self.bisection_atol = bisection_atol
        self.bisection_rtol = bisection_rtol
        self.max_iterations = max_iterations
        self.initial_guess = initial_guess
        self.lmi_strictness = lmi_strictness
        self.solver_params = solver_params
        self.n_jobs = n_jobs
        self.objective = objective

    def compute_ssv(
        self,
        N_omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
            if self.solver_params is None
            else self.solver_params
        )
        lmi_strictness = (
            utilities._auto_lmi_strictness(solver_params)
            if self.lmi_strictness is None
            else self.lmi_strictness
        )
        if self.objective == "minimize":
            constant_objective = False
        elif self.objective == "constant":
            constant_objective = True
        else:
            raise ValueError("`objective` must be `'minimize'` or `'constant'`.")

        def _ssv_at_omega(
            N_omega: np.ndarray,
        ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
            """Compute the structured singular value at a given frequency.

            Split into its own function to allow parallelization over
            frequencies with ``joblib``.

            Parameters
            ----------
            N_omega : np.ndarray
                Closed-loop tranfer function evaluated at a given frequency.

            Returns
            -------
            Tuple[float, np.ndarray, Dict[str, Any]]
                Structured singular value, D-scaling, and solution information.
                If the structured singular value cannot be computed, the first
                two elements of the tuple are ``None``, but solution
                information is still returned.

            Raises
            ------
            ValueError
                If ``objective`` is incorrectly set.
            """
            info = {}
            # Get an optimization variable that shares its block structure with
            # the D-scalings
            X = _variable_from_block_structure(block_structure)
            # Set objective function
            if constant_objective:
                objective = cvxpy.Minimize(1)
            else:
                # `X` should have a real diagonal because it's Hermitian, but
                # CVXPY does not realize this
                objective = cvxpy.Minimize(cvxpy.real(cvxpy.trace(X)))
            # Set upper bound on structured singular value squared as a parameter
            gamma_sq = cvxpy.Parameter(1, name="gamma_sq")
            # Set up the constraints
            constraints = [
                X.conj().T == X,
                X >> lmi_strictness,
                N_omega.conj().T @ X @ N_omega - gamma_sq * X << -lmi_strictness,
            ]
            # Put everything together in the optimization problem
            problem = cvxpy.Problem(objective, constraints)
            # Make sure initial guess is high enough
            gamma_high = self.initial_guess
            gammas = []
            problems = []
            results = []
            n_iterations = 0
            for i in range(self.max_iterations):
                n_iterations += 1
                gammas.append(gamma_high)
                try:
                    # Update gamma and solve optimization problem
                    problem.param_dict["gamma_sq"].value = np.array([gamma_high**2])
                    with warnings.catch_warnings():
                        # Ignore warnings since some problems may be infeasible
                        warnings.simplefilter("ignore")
                        result = problem.solve(**solver_params)
                        problems.append(problem)
                        results.append(result)
                except cvxpy.SolverError:
                    gamma_high *= 2
                    continue
                if isinstance(result, str) or (problem.status != "optimal"):
                    gamma_high *= 2
                else:
                    break
            else:
                info["status"] = "Could not find feasible initial `gamma`."
                info["gammas"] = gammas
                info["problems"] = problems
                info["results"] = results
                info["iterations"] = n_iterations
                return None, None, info
            # Check if ``gamma`` was increased.
            if gamma_high > self.initial_guess:
                warnings.warn(
                    f"Had to increase initial guess from {self.initial_guess} "
                    f"to {gamma_high}. Consider increasing the initial guess."
                )
            # Start iteration
            gamma_low = 0
            for i in range(self.max_iterations):
                n_iterations += 1
                gammas.append((gamma_high + gamma_low) / 2)
                try:
                    # Update gamma and solve optimization problem
                    problem.param_dict["gamma_sq"].value = np.array([gammas[-1] ** 2])
                    with warnings.catch_warnings():
                        # Ignore warnings since some problems may be infeasible
                        warnings.simplefilter("ignore")
                        result = problem.solve(**solver_params)
                        problems.append(problem)
                        results.append(result)
                except cvxpy.SolverError:
                    gamma_low = gammas[-1]
                    continue
                if isinstance(result, str) or (problem.status != "optimal"):
                    gamma_low = gammas[-1]
                else:
                    gamma_high = gammas[-1]
                    # Only terminate if last iteration succeeded to make sure
                    # ``X`` has a value
                    if np.isclose(
                        gamma_high,
                        gamma_low,
                        rtol=self.bisection_rtol,
                        atol=self.bisection_atol,
                    ):
                        break
            else:
                # Terminated due to max iterations
                info["status"] = "Reached maximum number of iterations."
                info["gammas"] = gammas
                info["problems"] = problems
                info["results"] = results
                info["iterations"] = n_iterations
                return None, None, info
            # Save info
            info["status"] = "Bisection succeeded."
            info["gammas"] = gammas
            info["problems"] = problems
            info["results"] = results
            info["iterations"] = n_iterations
            D_omega = scipy.linalg.cholesky(X.value)
            return (gammas[-1], D_omega, info)

        # Compute structured singular value and D scales for each frequency
        joblib_results = joblib.Parallel(n_jobs=self.n_jobs)(
            [
                joblib.delayed(_ssv_at_omega)(N_omega[:, :, i])
                for i in range(N_omega.shape[2])
            ]
        )
        # Extract and return results
        mu_lst, D_scales_lst, info_lst = tuple(zip(*joblib_results))
        mu = np.array(mu_lst)
        D_scales = np.moveaxis(np.array(D_scales_lst), 0, 2)
        info = {k: [d[k] for d in info_lst] for k in info_lst[0].keys()}
        info["solver_params"] = solver_params
        info["lmi_strictness"] = lmi_strictness
        info["constant_objective"] = constant_objective
        return mu, D_scales, info


def _variable_from_block_structure(block_structure: np.ndarray) -> cvxpy.Variable:
    """Get optimization variable with specified block structure.

    Parameters
    ----------
    block_structure : np.ndarray
        2D array with 2 columns and as many rows as uncertainty blocks
        in Delta. The columns represent the number of rows and columns in
        each uncertainty block.

    Returns
    -------
    cvxpy.Variable
        CVXPY variable with specified block structure.
    """
    X_lst = []
    for i in range(block_structure.shape[0]):
        row = []
        for j in range(block_structure.shape[0]):
            if i == j:
                # If on the block diagonal, insert variable
                if block_structure[i, 0] <= 0:
                    raise NotImplementedError(
                        "Real perturbations are not yet supported."
                    )
                if block_structure[i, 1] <= 0:
                    raise NotImplementedError(
                        "Diagonal perturbations are not yet supported."
                    )
                if block_structure[i, 0] != block_structure[i, 1]:
                    raise NotImplementedError(
                        "Nonsquare perturbations are not yet supported."
                    )
                if i == block_structure.shape[0] - 1:
                    # Last scaling is always identity
                    row.append(np.eye(block_structure[i, 0]))
                else:
                    # Every other scaling is either a scalar or a scalar
                    # multiplied by identity
                    if block_structure[i, 0] == 1:
                        xi = cvxpy.Variable((1, 1), complex=True, name=f"x{i}")
                        row.append(xi)
                    else:
                        xi = cvxpy.Variable(1, complex=True, name=f"x{i}")
                        row.append(xi * np.eye(block_structure[i, 0]))
            else:
                # If off the block diagonal, insert zeros
                row.append(np.zeros((block_structure[i, 0], block_structure[j, 1])))
        X_lst.append(row)
    X = cvxpy.bmat(X_lst)
    return X
