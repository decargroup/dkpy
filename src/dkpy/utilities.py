"""Transfer function and state-space manipulation utilities."""

from typing import Optional, Union, List

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


def _tf_combine(
    tf_array: List[List[Union[ArrayLike, control.TransferFunction]]],
) -> control.TransferFunction:
    """Combine array-like of transfer functions into MIMO transfer function.

    Parameters
    ----------
    tf_array : List[List[Union[ArrayLike, control.TransferFunction]]]
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
    ValueError
        If ``tf_array`` has incorrect dimensions.

    Examples
    --------
    >>> s = control.TransferFunction.s
    >>> dkpy._combine([
    ...     [1 / (s + 1)],
    ...     [s / (s + 2)],
    ... ])
    """
    # Find common timebase or raise error
    dt_list = []
    try:
        for row in tf_array:
            for tf in row:
                dt_list.append(getattr(tf, "dt", None))
    except OSError:
        raise ValueError("`tf_array` has too few dimensions.")
    dt_set = set(dt_list)
    dt_set.discard(None)
    if len(dt_set) > 1:
        raise ValueError(f"Timesteps of transfer functions are mismatched: {dt_set}")
    elif len(dt_set) == 0:
        dt = None
    else:
        dt = dt_set.pop()
    # Convert all entries to transfer function objects
    ensured_tf_array = []
    for row in tf_array:
        ensured_row = []
        for tf in row:
            ensured_row.append(_ensure_tf(tf, dt))
        ensured_tf_array.append(ensured_row)
    # Iterate over
    num = []
    den = []
    for row in ensured_tf_array:
        for j_out in range(row[0].noutputs):
            num_row = []
            den_row = []
            for col in row:
                for j_in in range(col.ninputs):
                    num_row.append(col.num[j_out][j_in])
                    den_row.append(col.den[j_out][j_in])
            num.append(num_row)
            den.append(den_row)
    G_tf = control.TransferFunction(num, den, dt=dt)
    return G_tf
