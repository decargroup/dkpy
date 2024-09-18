"""Transfer function and state-space manipulation utilities."""

import control
import numpy as np
from numpy.typing import ArrayLike


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
    bool :
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


def _tf_combine(tf_array: ArrayLike) -> control.TransferFunction:
    """Combine array-like of transfer functions into MIMO transfer function.

    Parameters
    ----------
    tf_array : ArrayLike
        Transfer matrix represented as a two-dimensional array or list-of-lists
        containing ``TransferFunction`` objects. The ``TransferFunction``
        objects can have multiple outputs and inputs, as long as the dimensions
        are compatible.

    Returns
    -------
    control.TransferFunction :
        Transfer matrix represented as a single MIMO ``TransferFunction``
        object.

    Examples
    --------
    >>> s = control.TransferFunction.s
    >>> dkpy._combine([
    ...     [1 / (s + 1)],
    ...     [s / (s + 2)],
    ... ])
    """
    # TODO Error checking
    # TODO Make sure scalars work
    # TODO check all timebases
    G = np.array(tf_array)
    num = []
    den = []
    # Iterate over rows and columns of transfer matrix
    for i_out in range(G.shape[0]):
        for j_out in range(G[i_out, 0].noutputs):
            num_row = []
            den_row = []
            # Iterate over rows and columns of inner `TransferFunction`
            for i_in in range(G.shape[1]):
                for j_in in range(G[i_out, i_in].ninputs):
                    num_row.append(G[i_out, i_in].num[j_out][j_in])
                    den_row.append(G[i_out, i_in].den[j_out][j_in])
            num.append(num_row)
            den.append(den_row)
    # Merge numerators and denominators into single transfer function
    G_tf = control.TransferFunction(num, den, dt=G[0][0].dt)
    return G_tf
