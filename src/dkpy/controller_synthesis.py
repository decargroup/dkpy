"""Controller synthesis classes."""

import abc
from typing import Any, Dict, Tuple

import control
import cvxpy
import numpy as np
import scipy.linalg


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
            information.
        """
        raise NotImplementedError()


class HinfSynSlicot(ControllerSynthesis):
    """H-infinity synthesis using SLICOT's Riccati equation method."""

    def __init__(self):
        """Instantiate :class:`HinfSynSlicot`."""
        pass

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        K, N, gamma, rcond = control.hinfsyn(P, n_y, n_u)
        info = {
            "rcond": rcond,
        }
        return K, N, gamma, info


class HinfSynLmi(ControllerSynthesis):
    """H-infinity synthesis using a linear matrix inequality approach."""

    def __init__(
        self,
        lmi_strictness: float = 1e-7,
    ):
        """Instantiate :class:`HinfSynLmi`."""
        self.lmi_strictness = lmi_strictness

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        info = {}
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
            X1 >> self.lmi_strictness,
            Y1 >> self.lmi_strictness,
            mat1 << -self.lmi_strictness,
            mat2 >> self.lmi_strictness,
        ]
        # Problem
        problem = cvxpy.Problem(objective, constraints)
        # Solver settings
        # TODO Allow these to be changed
        updated_solver_params = dict(
            solver="MOSEK",
            eps=1e-9,
            verbose=True,
        )
        # Solve problem
        # TODO Check solver status
        result = problem.solve(**updated_solver_params)
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
        # Extract block matrix of controller state-space matrices
        K_block = np.linalg.solve(M_right.T, np.linalg.solve(M_left, M_middle).T).T
        n_x_c = An.shape[0]
        A_K = K_block[:n_x_c, :n_x_c]
        B_K = K_block[:n_x_c, n_x_c:]
        C_K = K_block[n_x_c:, :n_x_c]
        D_K = K_block[n_x_c:, n_x_c:]
        # TODO Implement D22 != 0
        # Make sure ``D22`` is zero. If it's not, other calculations need to be
        # done to get ``K``.
        if not np.allclose(D22, np.zeros_like(D22), atol=1e-12, rtol=0):
            raise RuntimeError("TODO")
        # Create spate space object
        K = control.StateSpace(
            A_K,
            B_K,
            C_K,
            D_K,
        )
        N = P.lft(K)
        return K, N, gamma.value.item(), info
