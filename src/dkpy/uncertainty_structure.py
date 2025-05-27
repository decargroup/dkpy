"""Uncertainty block."""

import numpy as np
from typing import Sequence


class UncertaintyBlock:
    """Generic uncertainty block."""

    def __init__(
        self, num_inputs: int, num_outputs: int, is_diagonal: bool, is_complex: bool
    ):
        """Instantiate :class:`UncertaintyBlock`.

        Parameters
        ----------
        num_inputs : int
            Number of inputs (columns).
        num_outputs : int
            Number of outputs (rows).
        is_diagonal : bool
            Diagonality condition.
        is_diagonal : bool
            Complex-valued condition.
        """
        if num_inputs <= 0:
            raise ValueError("`num_inputs` must be greater than 0.")
        if num_outputs <= 0:
            raise ValueError("`num_outputs` must be greater than 0.")

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._is_diagonal = is_diagonal
        self._is_complex = is_complex

    @property
    def num_inputs(self):
        """Get number of inputs of the uncertainty block."""
        return self._num_inputs

    @property
    def num_outputs(self):
        """Get number of output of the uncertainty block."""
        return self._num_outputs

    @property
    def is_diagonal(self):
        """Get boolean for diagonal uncertainty."""
        return self._is_diagonal

    @property
    def is_complex(self):
        """Get boolean for complex uncertainty."""
        return self._is_complex


class RealDiagonalBlock(UncertaintyBlock):
    """Real-valued diagonal uncertainty block."""

    def __init__(self, num_channels: int):
        """Instantiate :class:`RealDiagonalBlock`.

        Parameters
        ----------
        num_channels : int
            Number of inputs/outputs to the uncertainty block.

        Raises
        ------
        ValueError
            If ``num_channels`` is not greater than zero.
        """
        if num_channels <= 0:
            raise ValueError("`num_channels` must be greater than 0.")

        super().__init__(
            num_inputs=num_channels,
            num_outputs=num_channels,
            is_diagonal=True,
            is_complex=False,
        )


class ComplexDiagonalBlock(UncertaintyBlock):
    """Complex-valued diagonal uncertainty block."""

    def __init__(self, num_channels: int):
        """Instantiate :class:`ComplexDiagonalBlock`.

        Parameters
        ----------
        num_channels : int
            Number of inputs/outputs to the uncertainty block.

        Raises
        ------
        ValueError
            If ``num_channels`` is not greater than zero.
        """
        if num_channels <= 0:
            raise ValueError("`num_channels` must be greater than 0.")

        super().__init__(
            num_inputs=num_channels,
            num_outputs=num_channels,
            is_diagonal=True,
            is_complex=True,
        )


class ComplexFullBlock(UncertaintyBlock):
    """Complex-valued full uncertainty block."""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Instantiate :class:`ComplexFullBlock`.

        Parameters
        ----------
        num_inputs : int
            Number of inputs (columns) to the uncertainty block.
        num_outputs : int
            Number of outputs (rows) to the uncertainty block.

        Raises
        ------
        ValueError
            If ``num_inputs`` is not greater than zero.
        ValueError
            If ``num_outputs`` is not greater than zero.
        """

        if num_inputs <= 0:
            raise ValueError("`num_inputs` must be greater than 0.")
        if num_outputs <= 0:
            raise ValueError("`num_outputs` must be greater than 0.")

        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            is_diagonal=False,
            is_complex=True,
        )


def _convert_matlab_block_structure(
    block_structure_matlab: np.ndarray,
) -> Sequence[UncertaintyBlock]:
    """
    Convert the MATLAB uncertainty block structure array description into the
    object-oriented uncertainty block structure description. The MATLAB uncertainty
    block structure description is given by
    - block_structure_matlab[i,:] = [-r 0]: i-th block is an r-by-r repeated, diagonal
      real scalar perturbation.
    - block_structure_matlab[i,:] = [r 0]: i-th block is an r-by-r repeated, diagonal
      complex scalar perturbation;
    - block_structure_matlab[i,:] = [r c]: i-th block is an r-by-c complex full-block
      perturbation.
    For additional reference, see [#matlab_block_struct]_.

    Parameters
    ----------
    block_structure_matlab : np.ndarray
        MATLAB uncertainty block structure representation.

    Returns
    -------
    Sequence[UncertaintyBlock]
        Object-oriented uncertainty block structure representation.

    Raises
    ------
    ValueError
        If the ``block_structure_matlab`` array does not have the correct shape.
    ValueError
        If a sub-array of ``block_structure_matlab`` is not a valid uncertainty block
        description.

    References
    ----------
    .. [#matlab_block_struct] https://www.mathworks.com/help/robust/ref/mussv.html
    """
    if block_structure_matlab.shape[1] != 2:
        raise ValueError(
            "Invalid MATLAB block structure array shape. The block structure array"
            f" shape must have `2` columns. However, it has {block_structure_matlab[1]}"
            " columns."
        )

    block_structure = []
    for block in block_structure_matlab:
        if (block[0] < 0) and (block[1] == 0):
            block_structure.append(RealDiagonalBlock(np.abs(block[0])))
        elif (block[0] > 0) and (block[1] == 0):
            block_structure.append(ComplexDiagonalBlock(block[0]))
        elif (block[0] > 0) and (block[1] > 0):
            block_structure.append(ComplexFullBlock(block[1], block[0]))
        else:
            raise ValueError("The uncertainty block array is invalid.")

    return block_structure
