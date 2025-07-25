"""Uncertainty block."""

import numpy as np
import abc
from typing import List, Union


class UncertaintyBlock(metaclass=abc.ABCMeta):
    """Generic uncertainty block."""

    @property
    @abc.abstractmethod
    def num_inputs(self) -> int:
        """Get number of inputs of the uncertainty block."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_outputs(self) -> int:
        """Get number of output of the uncertainty block."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_diagonal(self) -> bool:
        """Get boolean for diagonal uncertainty."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_complex(self) -> bool:
        """Get boolean for complex uncertainty."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_square(self) -> bool:
        """Get boolean for square uncertainty."""
        raise NotImplementedError()


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

        self._num_channels = num_channels

    @property
    def num_inputs(self) -> int:
        return self._num_channels

    @property
    def num_outputs(self) -> int:
        """Get number of output of the uncertainty block."""
        return self._num_channels

    @property
    def is_diagonal(self) -> bool:
        """Get boolean for diagonal uncertainty."""
        return True

    @property
    def is_complex(self) -> bool:
        """Get boolean for complex uncertainty."""
        return False

    @property
    def is_square(self) -> bool:
        """Get boolean for square uncertainty."""
        return True


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

        self._num_channels = num_channels

    @property
    def num_inputs(self) -> int:
        return self._num_channels

    @property
    def num_outputs(self) -> int:
        """Get number of output of the uncertainty block."""
        return self._num_channels

    @property
    def is_diagonal(self) -> bool:
        """Get boolean for diagonal uncertainty."""
        return True

    @property
    def is_complex(self) -> bool:
        """Get boolean for complex uncertainty."""
        return True

    @property
    def is_square(self) -> bool:
        """Get boolean for square uncertainty."""
        return True


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

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Get number of output of the uncertainty block."""
        return self._num_outputs

    @property
    def is_diagonal(self) -> bool:
        """Get boolean for diagonal uncertainty."""
        return False

    @property
    def is_complex(self) -> bool:
        """Get boolean for complex uncertainty."""
        return True

    @property
    def is_square(self) -> bool:
        """Get boolean for square uncertainty."""
        return self._num_inputs == self._num_outputs


def _convert_block_structure_representation(
    block_structure: Union[List[UncertaintyBlock], List[List[int]], np.ndarray],
) -> List[UncertaintyBlock]:
    """
    Convert the uncertainty block description into the object-oriented description.

    The MATLAB uncertainty block structure description is given by
    - block_structure_matlab[i,:] = [-r 0]: i-th block is an r-by-r repeated, diagonal
      real scalar perturbation.
    - block_structure_matlab[i,:] = [r 0]: i-th block is an r-by-r repeated, diagonal
      complex scalar perturbation.
    - block_structure_matlab[i,:] = [r c]: i-th block is an r-by-c complex full-block
      perturbation.
    For additional reference, see [#matlab_block_struct]_.

    Parameters
    ----------
    block_structure: Union[List[UncertaintyBlock], List[int]],
        Uncertainty block structure representation.

    Returns
    -------
    List[UncertaintyBlock]
        Object-oriented uncertainty block structure representation.

    Raises
    ------
    ValueError
        Block structure cannot be converted to an array.
    ValueError
        Block structure does not have the correct number of columns for the
        MATLAB uncertainty structure representation.
    ValueError
        Uncertainty block does not correspond to MATLAB standard description.

    References
    ----------
    .. [#matlab_block_struct] https://www.mathworks.com/help/robust/ref/mussv.html
    """

    # Block structure is in OOP representation
    is_oop_representation = all(
        isinstance(block, UncertaintyBlock) for block in block_structure
    )
    if is_oop_representation:
        return block_structure
    # Block structure is in MATLAB form
    block_structure_matlab = np.array(block_structure)
    if block_structure_matlab.shape[1] != 2:
        raise ValueError(
            "The `block_structure` array does not have the correct dimemsions "
            "for the MATLAB uncertainty specification. It must have 2 columns, "
            f"but it has {block_structure_matlab.shape[1]}."
        )
    block_structure_oop = []
    for block_matlab in block_structure_matlab:
        if (block_matlab[0] < 0) and (block_matlab[1] == 0):
            block_structure_oop.append(RealDiagonalBlock(np.abs(block_matlab[0])))
        elif (block_matlab[0] > 0) and (block_matlab[1] == 0):
            block_structure_oop.append(ComplexDiagonalBlock(block_matlab[0]))
        elif (block_matlab[0] > 0) and (block_matlab[1] > 0):
            block_structure_oop.append(
                ComplexFullBlock(block_matlab[1], block_matlab[0])
            )
        else:
            raise ValueError(
                "The uncertainty block array does not conform to the MATLAB standard."
            )

    return block_structure_oop
