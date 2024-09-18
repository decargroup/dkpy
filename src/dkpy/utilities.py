"""Transfer function and state-space manipulation utilities."""

import control

def _combine(tf_array):
    """Combine array-like of transfer functions into MIMO transfer function.

    Parameters
    ----------
    tf_array : 2D array_like of ``TransferFunction``s
        Transfer matrix represented as a two-dimensional array or list-of-lists
        containing ``TransferFunction`` objects. The ``TransferFunction``
        objects can have multiple outputs and inputs, as long as the dimensions
        are compatible.

    Returns
    -------
    control.TransferFunction :
        Transfer matrix represented as a single MIMO `TransferFunction` object.

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
