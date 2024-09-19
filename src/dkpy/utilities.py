"""Transfer function and state-space manipulation utilities."""

from typing import Optional, Union

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
    tf = control.TransferFunction(
        arraylike_3d,
        np.ones_like(arraylike_3d),
        dt,
    )
    return tf


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
    control.TransferFunction
        Transfer matrix represented as a single MIMO ``TransferFunction``
        object.

    Raises
    ------
    ValueError
        If timesteps of transfer functions do not match.

    Examples
    --------
    >>> s = control.TransferFunction.s
    >>> dkpy._combine([
    ...     [1 / (s + 1)],
    ...     [s / (s + 2)],
    ... ])
    """
    # Get set of unique timesteps in inputs
    tf_array_ = np.array(tf_array, dtype=object)
    dts = set([getattr(g, "dt", None) for g in tf_array_.ravel()])
    dts.discard(None)
    if len(dts) > 1:
        raise ValueError(f"Timesteps of transfer functions are mismatched: {dts}")
    elif len(dts) == 0:
        dt = None
    else:
        dt = dts.pop()
    # Convert everything into a transfer function object
    G = np.zeros_like(tf_array_, dtype=object)
    for i in range(tf_array_.shape[0]):
        for j in range(tf_array_.shape[1]):
            G[i, j] = _ensure_tf(tf_array_[i, j], dt)
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
    G_tf = control.TransferFunction(num, den, dt=dt)
    return G_tf
