"""Transfer function and state-space manipulation utilities."""

__all__ = [
    "example_scherer1997_p907",
    "example_skogestad2006_p325",
    "_ensure_tf",
    "_tf_close_coeff",
    "_auto_lmi_strictness",
]

from typing import Any, Dict, Union

import control
import cvxpy
import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike


def example_scherer1997_p907() -> Dict[str, Any]:
    """Add generalized plant from [SGC97]_, Example 7 (p. 907)."""
    # Process model
    A = np.array([[0, 10, 2], [-1, 1, 0], [0, 2, -5]])
    B1 = np.array([[1], [0], [1]])
    B2 = np.array([[0], [1], [0]])
    # Plant output
    C2 = np.array([[0, 1, 0]])
    D21 = np.array([[2]])
    D22 = np.array([[0]])
    # Hinf performance
    C1 = np.array([[1, 0, 0], [0, 0, 0]])
    D11 = np.array([[0], [0]])
    D12 = np.array([[0], [1]])
    # Dimensions
    n_y = 1
    n_u = 1
    # Create generalized plant
    B_gp = np.block([B1, B2])
    C_gp = np.block([[C1], [C2]])
    D_gp = np.block([[D11, D12], [D21, D22]])
    P = control.StateSpace(A, B_gp, C_gp, D_gp)
    out = {
        "P": P,
        "n_y": n_y,
        "n_u": n_u,
    }
    return out


def example_skogestad2006_p325() -> Dict[str, Any]:
    """Add generalized plant from [SP06]_, Table 8.1 (p. 325)."""
    # Plant
    G0 = np.array(
        [
            [87.8, -86.4],
            [108.2, -109.6],
        ]
    )
    G = control.append(
        control.TransferFunction([1], [75, 1]),
        control.TransferFunction([1], [75, 1]),
    ) * control.TransferFunction(
        G0.reshape(2, 2, 1),
        np.ones((2, 2, 1)),
    )
    # Weights
    Wp = 0.5 * control.append(
        control.TransferFunction([10, 1], [10, 1e-5]),
        control.TransferFunction([10, 1], [10, 1e-5]),
    )
    Wi = control.append(
        control.TransferFunction([1, 0.2], [0.5, 1]),
        control.TransferFunction([1, 0.2], [0.5, 1]),
    )
    G.name = "G"
    Wp.name = "Wp"
    Wi.name = "Wi"
    sum_w = control.summing_junction(
        inputs=["u_w", "u_G"],
        dimension=2,
        name="sum_w",
    )
    sum_del = control.summing_junction(
        inputs=["u_del", "u_u"],
        dimension=2,
        name="sum_del",
    )
    split = control.summing_junction(
        inputs=["u"],
        dimension=2,
        name="split",
    )
    P = control.interconnect(
        syslist=[G, Wp, Wi, sum_w, sum_del, split],
        connections=[
            ["G.u", "sum_del.y"],
            ["sum_del.u_u", "split.y"],
            ["sum_w.u_G", "G.y"],
            ["Wp.u", "sum_w.y"],
            ["Wi.u", "split.y"],
        ],
        inplist=["sum_del.u_del", "sum_w.u_w", "split.u"],
        outlist=["Wi.y", "Wp.y", "-sum_w.y"],
    )
    # Dimensions
    n_y = 2
    n_u = 2
    # Inverse-based controller
    K = (
        0.7
        * control.append(
            control.TransferFunction([75, 1], [1, 1e-5]),
            control.TransferFunction([75, 1], [1, 1e-5]),
        )
        * control.TransferFunction(
            scipy.linalg.inv(G0).reshape(2, 2, 1),
            np.ones((2, 2, 1)),
        )
    )
    out = {
        "P": P,
        "n_y": n_y,
        "n_u": n_u,
        "K": K,
    }
    return out


def _tf_close_coeff(
    tf_a: control.TransferFunction,
    tf_b: control.TransferFunction,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two transfer functions have close coefficients.

    Parameters
    ----------
    tf_a : control.TransferFunction
        First transfer function.
    tf_b : control.TransferFunction
        Second transfer function.
    rtol : float
        Relative tolerance for :func:`np.allclose`.
    atol : float
        Absolute tolerance for :func:`np.allclose`.

    Returns
    -------
    bool
        True if transfer function cofficients are all close.
    """
    # Check number of outputs and inputs
    if tf_a.noutputs != tf_b.noutputs:
        return False
    if tf_a.ninputs != tf_b.ninputs:
        return False
    # Check timestep
    if tf_a.dt != tf_b.dt:
        return False
    # Check coefficient arrays
    for i in range(tf_a.noutputs):
        for j in range(tf_a.ninputs):
            if not np.allclose(tf_a.num[i][j], tf_b.num[i][j], rtol=rtol, atol=atol):
                return False
            if not np.allclose(tf_a.den[i][j], tf_b.den[i][j], rtol=rtol, atol=atol):
                return False
    return True


def _ensure_tf(
    arraylike_or_tf: Union[ArrayLike, control.TransferFunction],
    dt: Union[None, bool, float] = None,
) -> control.TransferFunction:
    """Convert an array-like to a transfer function.

    Parameters
    ----------
    arraylike_or_tf : Union[ArrayLike, control.TransferFunction]
        Array-like or transfer function.
    dt : Union[None, bool, float]
        Timestep (s). Based on the ``control`` package, ``True`` indicates a
        discrete-time system with unspecified timestep, ``0`` indicates a
        continuous-time system, and ``None`` indicates a continuous- or
        discrete-time system with unspecified timestep. If ``None``, timestep
        is not validated.

    Returns
    -------
    control.TransferFunction
        Transfer function.

    Raises
    ------
    ValueError
        If input cannot be converted to a transfer function.
    ValueError
        If the timesteps do not match.
    """
    # If the input is already a transfer function, return it right away
    if isinstance(arraylike_or_tf, control.TransferFunction):
        # If timesteps don't match, raise an exception
        if (dt is not None) and (arraylike_or_tf.dt != dt):
            raise ValueError(
                f"`arraylike_or_tf.dt={arraylike_or_tf.dt}` does not match argument `dt={dt}`."
            )
        return arraylike_or_tf
    if np.ndim(arraylike_or_tf) > 2:
        raise ValueError(
            "Array-like must have less than two dimensions to be converted into a transfer function."
        )
    # If it's not, then convert it to a transfer function
    arraylike_3d = np.atleast_3d(arraylike_or_tf)
    try:
        tf = control.TransferFunction(
            arraylike_3d,
            np.ones_like(arraylike_3d),
            dt,
        )
    except TypeError:
        raise ValueError(
            "`arraylike_or_tf` must only contain array-likes or transfer functions."
        )
    return tf


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
                    # For conic problems
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_INFEAS", 1e-12),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8),
                    mosek_params.get("MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-8),
                    # For linear problems
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
