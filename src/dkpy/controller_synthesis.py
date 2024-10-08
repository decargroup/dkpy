"""Controller synthesis classes."""

__all__ = [
    "ControllerSynthesis",
    "HinfSynSlicot",
    "HinfSynLmi",
    "HinfSynLmiBisection",
]

import abc
from typing import Any, Dict, Optional, Tuple
import warnings

import control
import cvxpy
import numpy as np
import scipy.linalg
import slycot


class ControllerSynthesis(metaclass=abc.ABCMeta):
    """Controller synthesis base class."""

    @abc.abstractclassmethod
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
        """
        raise NotImplementedError()


class HinfSynSlicot(ControllerSynthesis):
    """H-infinity synthesis using SLICOT's Riccati equation method.

    TODO Add example
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
        try:
            K, N, gamma, rcond = control.hinfsyn(P, n_y, n_u)
        except slycot.exceptions.SlycotError:
            return None, None, None, info
        info["rcond"] = rcond
        return K, N, gamma, info


class HinfSynLmi(ControllerSynthesis):
    """H-infinity synthesis using a linear matrix inequality approach.

    TODO Add example

    TODO Add reference
    Caverly and Forbes 2024, Section 5.3.3
    """

    def __init__(
        self,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate :class:`HinfSynLmi`."""
        self.lmi_strictness = lmi_strictness
        self.solver_params = solver_params

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        info = {}
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
            _auto_lmi_strictness(solver_params)
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
        info["solver_stats"] = problem.solver_stats
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
        )
        N = P.lft(K)
        return K, N, gamma.value.item(), info


class HinfSynLmiBisection(ControllerSynthesis):
    """H-infinity synthesis using an LMI approach with bisection."""

    def __init__(
        self,
        bisection_atol: float = 1e-5,
        bisection_rtol: float = 1e-4,
        max_iterations: int = 100,
        initial_guess: float = 10,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate :class:`HinfSynLmiBisection`."""
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
            _auto_lmi_strictness(solver_params)
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
                # Only terminate if last iteration succeeded to make sure ``X``
                # has a value.
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
        )
        N = P.lft(K)
        return K, N, gamma.value.item(), info


def _auto_lmi_strictness(
    solver_params: Dict[str, Any],
    scale: float = 10,
) -> float:
    """Autoselect LMI strictness based on solver settings.

    Parameters
    ----------
    solver_params : Dict[str, Any]
        Arguments that would be passed to :func:`cvxpy.Problem.solve`.
    scale : float = 10
        LMI strictness is ``scale`` times larger than the largest solver
        tolerance.

    Returns
    -------
    float
        LMI strictness.

    Raises
    ------
    ValueError
        If the solver specified is not recognized by CVXPY.
    """
    if solver_params["solver"] == cvxpy.CLARABEL:
        tol = np.max(
            [
                solver_params.get("tol_gap_abs", 1e-8),
                solver_params.get("tol_feas", 1e-8),
                solver_params.get("tol_infeas_abs", 1e-8),
            ]
        )
    elif solver_params["solver"] == cvxpy.COPT:
        tol = np.max(
            [
                solver_params.get("AbsGap", 1e-6),
                solver_params.get("DualTol", 1e-6),
                solver_params.get("FeasTol", 1e-6),
            ]
        )
    elif solver_params["solver"] == cvxpy.MOSEK:
        if "mosek_params" in solver_params.keys():
            mosek_params = solver_params["mosek_params"]
            tol = np.max(
                [
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_INFEAS", 1e-12),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_TOL_DFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_TOL_INFEAS", 1e-10),
                    mosek_params.get("MSK_DPAR_INTPNT_TOL_MU_RED", 1e-16),
                    mosek_params.get("MSK_DPAR_INTPNT_TOL_PFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8),
                ]
            )
        else:
            # If neither ``mosek_params`` nor ``eps`` are set, default to 1e-8
            tol = solver_params.get("eps", 1e-8)
    elif solver_params["solver"] == cvxpy.CVXOPT:
        tol = np.max(
            [
                solver_params.get("abstol", 1e-7),
                solver_params.get("feastol", 1e-7),
            ]
        )
    elif solver_params["solver"] == cvxpy.SDPA:
        tol = solver_params.get("epsilonStar", 1e-7)
    elif solver_params["solver"] == cvxpy.SCS:
        tol = solver_params.get("eps", 1e-4)
    else:
        raise ValueError(
            f"Solver {solver_params['solver']} is not a CVXPY-supported SDP solver."
        )
    strictness = scale * tol
    return strictness
