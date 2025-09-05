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


def example_mackenroth2004_p430():
    """Add generalized plant from [M04]_, Section 14.1 (p. 430)."""
    # Airframe model
    A_af = np.array(
        [
            [-0.0869, 0, 0.039, -1],
            [-4.424, -1.184, 0, 0.335],
            [0, 1, 0, 0],
            [2.148, -0.021, 0, -0.228],
        ]
    )
    B1_af = np.array([[0.0869, 0], [4.424, 1.184], [0, -1], [-2.148, 0.021]])
    B2_af = np.array([[0.0223, 0], [0.547, 2.12], [0, 0], [-1.169, 0.065]])
    B_af = np.block([[B1_af, B2_af]])
    C_af = np.eye(4)
    D_af = np.zeros((4, 4))
    airframe = control.StateSpace(A_af, B_af, C_af, D_af, name="airframe")
    airframe.set_inputs(["beta_w", "p_w", "zeta", "xi"])
    airframe.set_outputs(["beta", "p", "phi", "r"])

    # Actuator model
    A_act = np.array(
        [[0, 1, 0, 0], [-1600, -56, 0, 0], [0, 0, 0, 1], [0, 0, -1600, -56]]
    )
    B_act = np.array([[0, 0], [1600, 0], [0, 0], [0, 1600]])
    C_act = np.eye(4)
    D_act = np.zeros((4, 2))
    actuator = control.StateSpace(A_act, B_act, C_act, D_act, name="actuator")
    actuator.set_inputs(["zeta_unc", "xi_unc"])
    actuator.set_outputs(["zeta", "rate_zeta", "xi", "rate_xi"])

    # Performance weight
    weight_perf_angle = control.TransferFunction([1, 1.5], [1.5, 0.015])
    weight_perf_rate = control.TransferFunction([1, 1.5], [10, 20])
    weight_perf = control.append(
        weight_perf_angle,
        weight_perf_angle,
        weight_perf_rate,
        weight_perf_rate,
        name="weight_perf",
    )
    weight_perf.set_inputs(["e_phi", "beta", "p", "r"])
    weight_perf.set_outputs(weight_perf.noutputs, "z_p")

    # Actuator weight
    weight_actu_angle = control.StateSpace([], [], [], [[1 / 20]])
    weight_actu_rate = control.StateSpace([], [], [], [[1 / 60]])
    weight_actu = control.append(
        weight_actu_angle,
        weight_actu_angle,
        weight_actu_rate,
        weight_actu_rate,
        name="weight_actu",
    )
    weight_actu.set_inputs(["zeta_c", "xi_c", "rate_zeta", "rate_xi"])
    weight_actu.set_outputs(weight_actu.noutputs, "z_u")

    # Disturbance weight
    weight_dist_channel = control.StateSpace([], [], [], [[1]])
    weight_dist = control.append(
        weight_dist_channel, weight_dist_channel, name="weight_dist"
    )
    weight_dist.set_inputs(["beta_w_nor", "p_w_nor"])
    weight_dist.set_outputs(["beta_w", "p_w"])

    # Reference weight
    weight_ref = control.TransferFunction([2.25], [1, 2.1, 2.25])
    weight_ref = control.ss(weight_ref, name="weight_ref")
    weight_ref.set_inputs("phi_ref_nor")
    weight_ref.set_outputs("phi_ref")

    # Noise weight
    weight_noise_channel = 0.0005 * control.TransferFunction([1 / 0.02, 1], [1, 1])
    weight_noise = control.append(
        weight_noise_channel,
        weight_noise_channel,
        weight_noise_channel,
        weight_noise_channel,
        name="weight_noise",
    )
    weight_noise.set_inputs(weight_noise.ninputs, "n_nor")
    weight_noise.set_outputs(weight_noise.noutputs, "n")

    # Input multiplicative uncertainty weight
    weight_unc_channel = control.TransferFunction([0.25, 0.05], [0.125, 1])
    weight_unc = control.append(
        weight_unc_channel,
        weight_unc_channel,
        name="weight_unc",
    )
    weight_unc.set_inputs(["zeta_c", "xi_c"])
    weight_unc.set_outputs(weight_unc.noutputs, "y_del")

    # Uncertainty summation junction
    sum_uncertainty = control.StateSpace([], [], [], [[1, 0, 1, 0], [0, 1, 0, 1]])
    sum_uncertainty.set_inputs(["zeta_c", "xi_c", "u_del[0]", "u_del[1]"])
    sum_uncertainty.set_outputs(["zeta_unc", "xi_unc"])

    # Reference tracking error difference junction
    sum_error = control.StateSpace([], [], [], [[-1, 1]])
    sum_error.set_inputs(["phi", "phi_ref"])
    sum_error.set_outputs("e_phi")

    # Noise summation junction
    sum_noise = control.StateSpace([], [], [], np.block([[np.eye(4), np.eye(4)]]))
    sum_noise.set_inputs(["phi", "beta", "p", "r", "n[0]", "n[1]", "n[2]", "n[3]"])
    sum_noise.set_outputs(["phi_meas", "beta_meas", "p_meas", "r_meas"])

    # Reference signal passthrough (control.interconnect cannot have the interconnected
    # system have an output that is not the output of a sub-system)
    passthrough_ref = control.StateSpace([], [], [], np.eye(1))
    passthrough_ref.set_inputs(["phi_ref_exo"])
    passthrough_ref.set_outputs(["phi_ref_nor"])

    # Generalized plant
    input_id_list = [
        "u_del[0]",
        "u_del[1]",
        "phi_ref_exo",
        "n_nor[0]",
        "n_nor[1]",
        "n_nor[2]",
        "n_nor[3]",
        "beta_w_nor",
        "p_w_nor",
        "zeta_c",
        "xi_c",
    ]
    output_id_list = [
        "y_del[0]",
        "y_del[1]",
        "z_p[0]",
        "z_p[1]",
        "z_p[2]",
        "z_p[3]",
        "z_u[0]",
        "z_u[1]",
        "z_u[2]",
        "z_u[3]",
        "phi_ref_nor",
        "phi_meas",
        "beta_meas",
        "p_meas",
        "r_meas",
    ]
    plant_gen = control.interconnect(
        syslist=[
            airframe,
            actuator,
            weight_perf,
            weight_actu,
            weight_dist,
            weight_ref,
            weight_noise,
            weight_unc,
            sum_uncertainty,
            sum_error,
            sum_noise,
            passthrough_ref,
        ],
        inplist=input_id_list,
        outlist=output_id_list,
        name="plant_gen",
    )
    plant_gen.set_inputs(input_id_list)
    plant_gen.set_outputs(output_id_list)
    plant_gen = plant_gen.minreal()
    # Dimensions
    n_y = 5
    n_u = 2
    n_u_delta = 2
    n_y_delta = 2
    n_w = 7
    n_z = 8
    # Example output dictionary
    out = {
        "P": plant_gen,
        "airframe": airframe,
        "actuator": actuator,
        "weight_unc": weight_unc,
        "sum_noise": sum_noise,
        "sum_uncertainty": sum_uncertainty,
        "n_u": n_u,
        "n_y": n_y,
        "n_u_delta": n_u_delta,
        "n_y_delta": n_y_delta,
        "n_w": n_w,
        "n_z": n_z,
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
