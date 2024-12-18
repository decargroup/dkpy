"Uncertainty bound identification."

import numpy as np
import scipy.linalg
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
        The uncertainty bound, a nonminimum-phase, asymptotically stable,
        bibroper, transfer function.
    """
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

    nom_resp: np.ndarray = None
    if nom_model.dt == 0:
        nom_resp = nom_model(1e0j * freq_rng)
    else:
        nom_resp = nom_model(np.exp(1e0j * freq_rng))

    off_nom_resp: np.ndarray = np.zeros(ny, nu, nw, nk)
    for k in range(nk):
        if nom_model.dt == 0:
            off_nom_resp[:, :, :, k] = off_nom_models[k](1e0j * freq_rng)
        else:
            off_nom_resp[:, :, :, k] = off_nom_models[k](np.exp(1e0j * freq_rng))

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

    # TODO Define the optimization variable

    # TODO Pose the problem objective

    # TODO Pose the problem constraints

    # TODO Form an initial guess

    # TODO Solve the optimization

    # TODO Extract the optimizer and form the upper bound filter
    raise NotImplementedError()
