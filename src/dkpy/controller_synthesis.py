"""Controller synthesis classes."""

__all__ = [
    "ControllerSynthesis",
    "HinfSynSlicot",
    "HinfSynLmi",
    "HinfSynLmiBisection",
]

import abc
import warnings
from typing import Any, Dict, Optional, Tuple

import control
import cvxpy
import numpy as np
import scipy.linalg
import slycot

from . import utilities


class ControllerSynthesis(metaclass=abc.ABCMeta):
    """Controller synthesis base class."""

    @abc.abstractmethod
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
            Controller, closed-loop system, objective function value, solution
            information. If a controller cannot by synthesized, the first three
            elements of the tuple are ``None``, but solution information is
            still returned.

        Raises
        ------
        ValueError
            If the solver specified is not recognized by CVXPY.
        ValueError
            If the generalized plant is not continuous-time
            (i.e., if ``p.dt != 0``).
        """
        raise NotImplementedError()


class HinfSynSlicot(ControllerSynthesis):
    """H-infinity synthesis using SLICOT's Riccati equation method.

    Examples
    --------
    H-infinity controller synthesis

    >>> P, n_y, n_u = example_scherer1997
    >>> K, N, gamma, info = dkpy.HinfSynSlicot().synthesize(P, n_y, n_u)
    """

    def __init__(self):
        """Instantiate :class:`HinfSynSlicot`."""
        pass

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        info = {}
        if P.dt != 0:
            raise ValueError("Generalized plant must be continuous-time (`P.dt = 0`).")
        try:
            K, N, gamma, rcond = control.hinfsyn(P, n_y, n_u)
        except slycot.exceptions.SlycotError:
            return None, None, None, info
        info["rcond"] = rcond
        return K, N, gamma, info


class HinfSynLmi(ControllerSynthesis):
    """H-infinity synthesis using a linear matrix inequality approach.

    Synthesis method based on Section 5.3.3 of [CF24]_.

    Examples
    --------
    H-infinity controller synthesis with default settings

    >>> P, n_y, n_u = example_scherer1997
    >>> K, N, gamma, info = dkpy.HinfSynLmi().synthesize(P, n_y, n_u)

    H-infinity controller synthesis with CLARABEL

    >>> K, N, gamma, info = dkpy.HinfSynLmi(
    ...     lmi_strictness=1e-8,
    ...     solver_params={
    ...         "solver": "CLARABEL",
    ...         "tol_gap_abs": 1e-9,
    ...         "tol_gap_rel": 1e-9,
    ...         "tol_feas": 1e-9,
    ...         "tol_infeas_abs": 1e-9,
    ...         "tol_infeas_rel": 1e-9,
    ...     },
    ... ).synthesize(P, n_y, n_u)

    H-infinity controller synthesis with SCS

    >>> K, N, gamma, info = dkpy.HinfSynLmi(
    ...     lmi_strictness=1e-3,
    ...     solver_params={
    ...         "solver": "SCS",
    ...         "eps": 1e-4,
    ...     },
    ... ).synthesize(P, n_y, n_u)

    H-infinity controller synthesis with MOSEK (simple settings)

    >>> K, N, gamma, info = dkpy.HinfSynLmi(
    ...     lmi_strictness=1e-8,
    ...     solver_params={
    ...         "solver": "MOSEK",
    ...         "eps": 1e-9,
    ...     },
    ... ).synthesize(P, n_y, n_u)  # doctest: +SKIP

    H-infinity controller synthesis with MOSEK (advanced settings)

    >>> K, N, gamma, info = dkpy.HinfSynLmi(
    ...     lmi_strictness=1e-7,
    ...     solver_params={
    ...         "solver": "MOSEK",
    ...         "mosek_params": {
    ...             "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
    ...             "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
    ...             "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-8,
    ...             "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
    ...             "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
    ...         },
    ...     },
    ... ).synthesize(P, n_y, n_u)  # doctest: +SKIP
    """

    def __init__(
        self,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate :class:`HinfSynLmi`.

        Solution accuracy depends strongly on the selected solver and
        tolerances. Setting the solver and its tolerances in ``solver_params``
        and setting ``lmi_strictness`` manually is recommended, rather than
        relying on the default settings.

        Parameters
        ----------
        lmi_strictness : Optional[float]
            Strictness for linear matrix inequality constraints. Should be
            larger than the solver tolerance. If ``None``, then it is
            automatically set to 10x the solver's largest absolute tolerance.
        solver_params : Optional[Dict[str, Any]]
            Dictionary of keyword arguments for :func:`cvxpy.Problem.solve`.
            Notable keys are ``'solver'`` and ``'verbose'``. Additional keys
            used to set solver tolerances are solver-dependent. A definitive
            list can be found at [#solvers]_.

        References
        ----------
        .. [#solvers] https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        """
        self.lmi_strictness = lmi_strictness
        self.solver_params = solver_params

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        info = {}
        if P.dt != 0:
            raise ValueError("Generalized plant must be continuous-time (`P.dt = 0`).")
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
        info["solver_params"] = solver_params
        lmi_strictness = (
            utilities._auto_lmi_strictness(solver_params)
            if self.lmi_strictness is None
            else self.lmi_strictness
        )
        info["lmi_strictness"] = lmi_strictness
        # Constants
        n_x = P.nstates
        n_w = P.ninputs - n_u
        n_z = P.noutputs - n_y
        A = P.A
        B1 = P.B[:, :n_w]
        B2 = P.B[:, n_w:]
        C1 = P.C[:n_z, :]
        C2 = P.C[n_z:, :]
        D11 = P.D[:n_z, :n_w]
        D12 = P.D[:n_z, n_w:]
        D21 = P.D[n_z:, :n_w]
        D22 = P.D[n_z:, n_w:]
        # Variables
        An = cvxpy.Variable((n_x, n_x), name="An")
        Bn = cvxpy.Variable((n_x, n_y), name="Bn")
        Cn = cvxpy.Variable((n_u, n_x), name="Cn")
        Dn = cvxpy.Variable((n_u, n_y), name="Dn")
        X1 = cvxpy.Variable((n_x, n_x), name="X1", symmetric=True)
        Y1 = cvxpy.Variable((n_x, n_x), name="Y1", symmetric=True)
        gamma = cvxpy.Variable(1, name="gamma")
        # Objective
        objective = cvxpy.Minimize(gamma)
        # Constraints
        mat1 = cvxpy.bmat(
            [
                [
                    A @ Y1 + Y1.T @ A.T + B2 @ Cn + Cn.T @ B2.T,
                    A + An.T + B2 @ Dn @ C2,
                    B1 + B2 @ Dn @ D21,
                    Y1.T @ C1.T + Cn.T @ D12.T,
                ],
                [
                    (A + An.T + B2 @ Dn @ C2).T,
                    X1 @ A + A.T @ X1.T + Bn @ C2 + C2.T @ Bn.T,
                    X1 @ B1 + Bn @ D21,
                    C1.T + C2.T @ Dn.T @ D12.T,
                ],
                [
                    (B1 + B2 @ Dn @ D21).T,
                    (X1 @ B1 + Bn @ D21).T,
                    cvxpy.multiply(-gamma, np.eye(D11.shape[1])),
                    D11.T + D21.T @ Dn.T @ D12.T,
                ],
                [
                    (Y1.T @ C1.T + Cn.T @ D12.T).T,
                    (C1.T + C2.T @ Dn.T @ D12.T).T,
                    (D11.T + D21.T @ Dn.T @ D12.T).T,
                    cvxpy.multiply(-gamma, np.eye(D11.shape[0])),
                ],
            ]
        )
        mat2 = cvxpy.bmat(
            [
                [X1, np.eye(X1.shape[0])],
                [np.eye(Y1.shape[0]), Y1],
            ]
        )
        constraints = [
            gamma >= 0,
            X1 >> lmi_strictness,
            Y1 >> lmi_strictness,
            mat1 << -lmi_strictness,
            mat2 >> lmi_strictness,
        ]
        # Problem
        problem = cvxpy.Problem(objective, constraints)
        # Solve problem
        result = problem.solve(**solver_params)
        info["result"] = result
        info["problem"] = problem
        if isinstance(result, str) or (problem.status != "optimal"):
            return None, None, None, info
        # Extract controller
        Q, s, Vt = scipy.linalg.svd(
            np.eye(X1.shape[0]) - X1.value @ Y1.value,
            full_matrices=True,
        )
        X2 = Q @ np.diag(np.sqrt(s))
        Y2 = Vt.T @ np.diag(np.sqrt(s))
        M_left = np.block(
            [
                [
                    X2,
                    X1.value @ B2,
                ],
                [
                    np.zeros((B2.shape[1], X2.shape[1])),
                    np.eye(B2.shape[1]),
                ],
            ]
        )
        M_middle = np.block(
            [
                [An.value, Bn.value],
                [Cn.value, Dn.value],
            ]
        ) - np.block(
            [
                [X1.value @ A @ Y1.value, np.zeros_like(Bn.value)],
                [np.zeros_like(Cn.value), np.zeros_like(Dn.value)],
            ]
        )
        M_right = np.block(
            [
                [
                    Y2.T,
                    np.zeros((Y2.T.shape[0], C2.shape[0])),
                ],
                [
                    C2 @ Y1.value,
                    np.eye(C2.shape[0]),
                ],
            ]
        )
        # Save condition numbers before inverting
        info["cond_M_left"] = np.linalg.cond(M_left)
        info["cond_M_right"] = np.linalg.cond(M_right)
        # Extract ``A_K``, ``B_K``, ``C_K``, and ``D_K``. If ``D22=0``, these
        # are the controller state-space matrices. If not, there is one more
        # step to do.
        K_block = scipy.linalg.solve(
            M_right.T, scipy.linalg.solve(M_left, M_middle).T
        ).T
        n_x_c = An.shape[0]
        A_K = K_block[:n_x_c, :n_x_c]
        B_K = K_block[:n_x_c, n_x_c:]
        C_K = K_block[n_x_c:, :n_x_c]
        D_K = K_block[n_x_c:, n_x_c:]
        # Compute controller state-space matrices if ``D22`` is nonzero.
        if np.any(D22):
            D_c = scipy.linalg.solve(np.eye(D_K.shape[0]) + D_K @ D22, D_K)
            C_c = (np.eye(D_c.shape[0]) - D_c @ D22) @ C_K
            B_c = B_K @ (np.eye(D22.shape[0]) - D22 @ D_c)
            A_c = A_K - B_c @ scipy.linalg.solve(
                np.eye(D22.shape[0]) - D22 @ D_c, D22 @ C_c
            )
        else:
            D_c = D_K
            C_c = C_K
            B_c = B_K
            A_c = A_K
        # Create spate space object
        K = control.StateSpace(
            A_c,
            B_c,
            C_c,
            D_c,
            dt=0,
        )
        N = P.lft(K)
        return K, N, gamma.value.item(), info


class HinfSynLmiBisection(ControllerSynthesis):
    """H-infinity synthesis using an LMI approach with bisection.

    Synthesis method based on Section 5.3.3 of [CF24]_.

    Examples
    --------
    H-infinity controller synthesis with default settings

    >>> P, n_y, n_u = example_scherer1997
    >>> K, N, gamma, info = dkpy.HinfSynLmiBisection().synthesize(P, n_y, n_u)

    H-infinity controller synthesis with CLARABEL

    >>> K, N, gamma, info = dkpy.HinfSynLmiBisection(
    ...     bisection_atol=1e-4,
    ...     bisection_rtol=1e-3,
    ...     max_iterations=20,
    ...     initial_guess=10,
    ...     lmi_strictness=1e-8,
    ...     solver_params={
    ...         "solver": "CLARABEL",
    ...         "tol_gap_abs": 1e-9,
    ...         "tol_gap_rel": 1e-9,
    ...         "tol_feas": 1e-9,
    ...         "tol_infeas_abs": 1e-9,
    ...         "tol_infeas_rel": 1e-9,
    ...     },
    ... ).synthesize(P, n_y, n_u)
    """

    def __init__(
        self,
        bisection_atol: float = 1e-5,
        bisection_rtol: float = 1e-4,
        max_iterations: int = 100,
        initial_guess: float = 10,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate :class:`HinfSynLmiBisection`.

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

        References
        ----------
        .. [#solvers] https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        """
        self.bisection_atol = bisection_atol
        self.bisection_rtol = bisection_rtol
        self.max_iterations = max_iterations
        self.initial_guess = initial_guess
        self.lmi_strictness = lmi_strictness
        self.solver_params = solver_params

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        info = {}
        if P.dt != 0:
            raise ValueError("Generalized plant must be continuous-time (`P.dt = 0`).")
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
        solver_params["warm_start"] = True  # Force warm start for bisection
        info["solver_params"] = solver_params
        lmi_strictness = (
            utilities._auto_lmi_strictness(solver_params)
            if self.lmi_strictness is None
            else self.lmi_strictness
        )
        info["lmi_strictness"] = lmi_strictness
        # Constants
        n_x = P.nstates
        n_w = P.ninputs - n_u
        n_z = P.noutputs - n_y
        A = P.A
        B1 = P.B[:, :n_w]
        B2 = P.B[:, n_w:]
        C1 = P.C[:n_z, :]
        C2 = P.C[n_z:, :]
        D11 = P.D[:n_z, :n_w]
        D12 = P.D[:n_z, n_w:]
        D21 = P.D[n_z:, :n_w]
        D22 = P.D[n_z:, n_w:]
        # Variables
        An = cvxpy.Variable((n_x, n_x), name="An")
        Bn = cvxpy.Variable((n_x, n_y), name="Bn")
        Cn = cvxpy.Variable((n_u, n_x), name="Cn")
        Dn = cvxpy.Variable((n_u, n_y), name="Dn")
        X1 = cvxpy.Variable((n_x, n_x), name="X1", symmetric=True)
        Y1 = cvxpy.Variable((n_x, n_x), name="Y1", symmetric=True)
        # Bisection parameter
        gamma = cvxpy.Parameter(1, name="gamma")
        # Constant objective since this is a feasibility problem for ``gamma``
        objective = cvxpy.Minimize(1)
        # Constraints
        mat1 = cvxpy.bmat(
            [
                [
                    A @ Y1 + Y1.T @ A.T + B2 @ Cn + Cn.T @ B2.T,
                    A + An.T + B2 @ Dn @ C2,
                    B1 + B2 @ Dn @ D21,
                    Y1.T @ C1.T + Cn.T @ D12.T,
                ],
                [
                    (A + An.T + B2 @ Dn @ C2).T,
                    X1 @ A + A.T @ X1.T + Bn @ C2 + C2.T @ Bn.T,
                    X1 @ B1 + Bn @ D21,
                    C1.T + C2.T @ Dn.T @ D12.T,
                ],
                [
                    (B1 + B2 @ Dn @ D21).T,
                    (X1 @ B1 + Bn @ D21).T,
                    cvxpy.multiply(-gamma, np.eye(D11.shape[1])),
                    D11.T + D21.T @ Dn.T @ D12.T,
                ],
                [
                    (Y1.T @ C1.T + Cn.T @ D12.T).T,
                    (C1.T + C2.T @ Dn.T @ D12.T).T,
                    (D11.T + D21.T @ Dn.T @ D12.T).T,
                    cvxpy.multiply(-gamma, np.eye(D11.shape[0])),
                ],
            ]
        )
        mat2 = cvxpy.bmat(
            [
                [X1, np.eye(X1.shape[0])],
                [np.eye(Y1.shape[0]), Y1],
            ]
        )
        constraints = [
            X1 >> lmi_strictness,
            Y1 >> lmi_strictness,
            mat1 << -lmi_strictness,
            mat2 >> lmi_strictness,
        ]
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
                problem.param_dict["gamma"].value = np.array([gamma_high])
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
            return None, None, None, info
        # Check if ``gamma`` was increased.
        if gamma_high > self.initial_guess:
            warnings.warn(
                f"Had to increase initial guess from {self.initial_guess} to {gamma_high}. "
                "Consider increasing the initial guess."
            )
        # Start iteration
        gamma_low = 0
        for i in range(self.max_iterations):
            n_iterations += 1
            gammas.append((gamma_high + gamma_low) / 2)
            try:
                # Update gamma and solve optimization problem
                problem.param_dict["gamma"].value = np.array([gammas[-1]])
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
                # variables have values
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
            return None, None, None, info
        # Save info
        info["status"] = "Bisection succeeded."
        info["gammas"] = gammas
        info["problems"] = problems
        info["results"] = results
        info["iterations"] = n_iterations
        # Extract controller
        Q, s, Vt = scipy.linalg.svd(
            np.eye(X1.shape[0]) - X1.value @ Y1.value,
            full_matrices=True,
        )
        X2 = Q @ np.diag(np.sqrt(s))
        Y2 = Vt.T @ np.diag(np.sqrt(s))
        M_left = np.block(
            [
                [
                    X2,
                    X1.value @ B2,
                ],
                [
                    np.zeros((B2.shape[1], X2.shape[1])),
                    np.eye(B2.shape[1]),
                ],
            ]
        )
        M_middle = np.block(
            [
                [An.value, Bn.value],
                [Cn.value, Dn.value],
            ]
        ) - np.block(
            [
                [X1.value @ A @ Y1.value, np.zeros_like(Bn.value)],
                [np.zeros_like(Cn.value), np.zeros_like(Dn.value)],
            ]
        )
        M_right = np.block(
            [
                [
                    Y2.T,
                    np.zeros((Y2.T.shape[0], C2.shape[0])),
                ],
                [
                    C2 @ Y1.value,
                    np.eye(C2.shape[0]),
                ],
            ]
        )
        # Save condition numbers before inverting
        info["cond_M_left"] = np.linalg.cond(M_left)
        info["cond_M_right"] = np.linalg.cond(M_right)
        # Extract ``A_K``, ``B_K``, ``C_K``, and ``D_K``. If ``D22=0``, these
        # are the controller state-space matrices. If not, there is one more
        # step to do.
        K_block = scipy.linalg.solve(
            M_right.T, scipy.linalg.solve(M_left, M_middle).T
        ).T
        n_x_c = An.shape[0]
        A_K = K_block[:n_x_c, :n_x_c]
        B_K = K_block[:n_x_c, n_x_c:]
        C_K = K_block[n_x_c:, :n_x_c]
        D_K = K_block[n_x_c:, n_x_c:]
        # Compute controller state-space matrices if ``D22`` is nonzero.
        if np.any(D22):
            D_c = scipy.linalg.solve(np.eye(D_K.shape[0]) + D_K @ D22, D_K)
            C_c = (np.eye(D_c.shape[0]) - D_c @ D22) @ C_K
            B_c = B_K @ (np.eye(D22.shape[0]) - D22 @ D_c)
            A_c = A_K - B_c @ scipy.linalg.solve(
                np.eye(D22.shape[0]) - D22 @ D_c, D22 @ C_c
            )
        else:
            D_c = D_K
            C_c = C_K
            B_c = B_K
            A_c = A_K
        # Create spate space object
        K = control.StateSpace(
            A_c,
            B_c,
            C_c,
            D_c,
            dt=0,
        )
        N = P.lft(K)
        return K, N, gamma.value.item(), info
