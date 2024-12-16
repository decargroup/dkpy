"Uncertainty bound identification."

import numpy as np
import control

from typing import List


def _identify_uncertainty_upper_bound(
    nom_model: control.LTI,
    off_nom_models: List[control.LTI],
    unc_struct: str,
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
    unc_struct : {'a', 'im', 'om', 'ia', 'iim', 'iom'}
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

    # Form the gain upper bound array
    # TODO Form the residuals depending on the chosen uncertainty structure
    # TODO Compute their frequency responses
    # TODO Compute their gain responses, i.e., their m.s.v. responses

    # TODO Define the optimization variable

    # TODO Pose the problem objective

    # TODO Pose the problem constraints

    # TODO Form an initial guess

    # TODO Solve the optimization

    # TODO Extract the optimizer and form the upper bound filter
    raise NotImplementedError()
