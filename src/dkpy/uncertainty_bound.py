"Uncertainty bound identification."

__all__ = [
    "_identify_uncertainty_upper_bound",
]

import numpy as np
import scipy.linalg
import scipy.optimize
import control

from typing import List


def _identify_uncertainty_upper_bound(
    nom_model: control.LTI,
    off_nom_models: List[control.LTI],
    unc_str: str,
    freq_rng: np.ndarray,
    order: int,
) -> control.TransferFunction:
    """Compute the uncertainty bound transfer function of off-nominal models.

    Parameters
    ----------
    nom_model : control.LTI
        The nominal model.
    off_nom_models : List[control.LTI]
        A list of the off-nominal models. Must be of the same shape as
        nom_model.
    unc_str : {'a', 'im', 'om', 'ia', 'iim', 'iom'}
        The uncertainty structure with respect to which to identify an
        uncertainty bound. One of the following options.
        'a' : Additive uncertainty
        'mi' : Multiplicative input uncertainty
        'mo' : Multiplicative output uncertainty
        'ia' : Inverse additive uncertainty
        'imi' : Inverse multiplicative input uncertainty
        'imo' : Inverse multiplicative output uncertainty
    freq_rng : np.ndarray
        The frequency range over which to optimize the shape of the uncertainty
        bound. In units of radians per second.
    order : int
        The order of the bibproper linear SISO filter that is the uncertainty
        bound.

    Returns
    -------
    w_E : control.TransferFunction
        The uncertainty bound: a nonminimum-phase, asymptotically stable,
        bibroper transfer function of given order.
    """
    # Form frequency response argument
    freq_arg: np.ndarray = None
    if nom_model.dt == 0:
        freq_arg = 1e0j * freq_rng
    else:
        raise NotImplementedError("Discrete time models are not yet handled.")
        freq_arg = np.exp(1e0j * freq_rng)

    # TODO Add checks to confirm that all user-input parameters are ok
    if unc_str not in {"a", "mi", "mo", "ia", "imi", "imo"}:
        raise ValueError("Invalid `unc_str` argument.")

    ny: int = nom_model.noutputs
    nu: int = nom_model.ninputs
    nk: int = len(off_nom_models)
    nw: int = np.size(freq_rng)

    def _form_residual_response(
        nom: np.ndarray,
        off: np.ndarray,
        unc_str: str,
    ) -> np.ndarray:
        """Compute the residual frequency response matrix."""
        res = None
        if unc_str == "a":
            res = off - nom
        elif unc_str == "mi":
            res = scipy.linalg.solve(nom, off - nom)
        elif unc_str == "mo":
            res = scipy.linalg.solve(nom.H, (off - nom).H).H
        elif unc_str == "ia":
            pre = scipy.linalg.solve(off, off - nom)
            res = scipy.linalg.solve(nom.H, pre.H).H
        elif unc_str == "imi":
            res = scipy.linalg.solve(off, off - nom)
        elif unc_str == "imo":
            res = scipy.linalg.solve(off.H, (off - nom).H).H
        return res

    nom_resp: np.ndarray = nom_model(freq_arg)

    off_nom_resp: np.ndarray = np.zeros(ny, nu, nw, nk)
    for k in range(nk):
        off_nom_resp[:, :, :, k] = off_nom_models[k](freq_arg)

    # TODO Improve the block below with numpy.vectorize
    res_resp: np.ndarray = np.zeros(ny, nu, nw, nk)
    for k in range(nk):
        for w in range(nw):
            res_resp[:, :, w, k] = _form_residual_response(
                nom_resp[:, :, w],
                off_nom_resp[:, :, w, k],
                unc_str,
            )

    # TODO Improve the below block with numpy.vectorize
    res_msv_resp: np.ndarray = np.zeros(nw, nk)
    for k in range(nk):
        for w in range(nw):
            s = scipy.linalg.svd(a=res_resp[:, :, w, k], compute_uv=False)
            res_msv_resp[w, k] = s[0]

    res_msv_resp_ub: np.ndarray = np.max(a=res_msv_resp, axis=1)

    # Pose the problem objective
    def _pre_J(x: np.ndarray) -> np.ndarray:
        """Compute the array of gain errors over freq_rng.

        The gain errors are the differences between the gain of w_E at guess x
        and the residual maximum singular value response upper bound.

        Parameters
        ----------
        x : np.ndarray
            Coefficients of the biproper rational function of given order that
            is the uncertainty weight. If order is equal to n and
                x = [a_n, ..., a_0, b_n, ..., b_0],
            then
                w_E(x) = a(s) / b(s),
            where
                a(s) = a_n s^n + ... + a_1 s + a_0
            and b(s) is similarly defined. Must be of size 2 * (order + 1).

        Returns
        -------
        err : np.ndarray
            The array of gain errors.
        """
        num_coeff: np.ndarray = x[: order + 1]
        num_eval = np.polyval(num_coeff, freq_arg)

        den_coeff: np.ndarray = x[-(order + 1) :]
        den_eval = np.polyval(den_coeff, freq_arg)

        w_E_gain = np.abs(num_eval / den_eval)

        err = w_E_gain - res_msv_resp_ub
        return err

    def _J(x: np.ndarray) -> float:
        """Compute the objective at guess x.

        Parameters
        ----------
        x : np.ndarray
            Coefficients of the biproper rational function of given order that
            is the uncertainty weight. If order is equal to n and
                x = [a_n, ..., a_0, b_n, ..., b_0],
            then
                w_E(x) = a(s) / b(s),
            where
                a(s) = a_n s^n + ... + a_1 s + a_0
            and b(s) is similarly defined. Must be of size 2 * (order + 1).

        Returns
        -------
        J : float
            The optimization objective, defined as the sum over all points in
            freq_rng of the square of the difference in gain between w_E and the
            m.s.v. of the residuals.
        """
        err = _pre_J(x)
        J = np.sum(err**2)
        return J

    # Pose the problem constraints
    constraint = {"type": "ineq", "fun": _pre_J}

    # Form an initial guess (constant gain at the peak of res_msv_resp_ub)
    x0 = np.zeros(2 * (order + 1))
    x0[order] = np.max(res_msv_resp_ub) + 1e-6
    x0[-1] = 1e0

    # Solve the optimization problem
    result = scipy.optimize.minimize(
        fun=_J,
        x0=list(x0),
        constraints=constraint,
        # options={'maxiter': 1000},
    )
    x_opt = result.x

    # Enforce NMP property (This only works for the CT case)
    num_coeff_opt = x_opt[: order + 1]
    num_roots = np.roots(num_coeff_opt)
    new_num_roots = -np.abs(np.real(num_roots)) + 1e0j * np.imag(num_roots)

    # Enforce the AS property (this only works for the CT case)
    den_coeff_opt = x_opt[-(order + 1) :]
    den_roots = np.roots(den_coeff_opt)
    new_den_roots = -np.abs(np.real(den_roots)) + 1e0j * np.imag(den_roots)

    # Extract the gain of the optimal filter
    gain = num_coeff_opt[0] / den_coeff_opt[0]

    # Form the filter
    w_E_opt = control.zpk(
        zeros=new_num_roots,
        poles=new_den_roots,
        gain=gain,
    )

    w_E_opt_minreal = control.minreal(w_E_opt, verbose=False)
    return w_E_opt_minreal
