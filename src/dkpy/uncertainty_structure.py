"""Uncertainty block."""

import numpy as np
import abc
import scipy
import cvxpy
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


class UncertaintyBlockStructure:
    def __init__(self, block_structure):
        self.block_list = self.convert_block_structure_representation(block_structure)

    def convert_block_structure_representation(
        self,
        block_structure: Union[List[UncertaintyBlock], np.typing.ArrayLike],
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
        block_structure: Union[List[UncertaintyBlock], np.typing.ArrayLike],
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
        if isinstance(block_structure, List):
            for block in block_structure:
                if not isinstance(block, UncertaintyBlock):
                    break
            else:
                return block_structure
        # Block structure is in MATLAB form
        try:
            block_structure_matlab = np.array(block_structure)
        except ValueError:
            raise ValueError(
                "The `block_structure` object cannot be converted to an array."
            )
        if block_structure_matlab.shape[1] != 2:
            raise ValueError(
                "The `block_structure` array does not have the correct dimemsions. "
                f"It must have 2 columns, but it has {block_structure_matlab.shape[1]}."
            )
        block_structure = []
        for block_matlab in block_structure_matlab:
            if (block_matlab[0] < 0) and (block_matlab[1] == 0):
                block_structure.append(RealDiagonalBlock(np.abs(block_matlab[0])))
            elif (block_matlab[0] > 0) and (block_matlab[1] == 0):
                block_structure.append(ComplexDiagonalBlock(block_matlab[0]))
            elif (block_matlab[0] > 0) and (block_matlab[1] > 0):
                block_structure.append(
                    ComplexFullBlock(block_matlab[1], block_matlab[0])
                )
            else:
                raise ValueError(
                    "The uncertainty block array does not conform to the MATLAB standard."
                )

        return block_structure

    def generate_d_scale_mask(self) -> np.ndarray:
        """Create a mask for the D-scale fit for the block structure.

        Entries known to be zero are set to 0. Entries known to be one are set to
        1. Entries to be fit numerically are set to -1.

        Returns
        -------
        np.ndarray
            Array of integers indicating zero, one, and unknown elements in the
            block structure.
        """
        num_blocks = len(self.block_list)
        X_lst = []
        for i in range(num_blocks):
            # Uncertainty block
            block = self.block_list[i]
            if not block.is_complex:
                raise NotImplementedError("Real perturbations are not yet supported.")
            if block.is_diagonal:
                raise NotImplementedError(
                    "Diagonal perturbations are not yet supported."
                )
            if not block.is_square:
                raise NotImplementedError(
                    "Nonsquare perturbations are not yet supported."
                )
            # Set last scaling to identity
            if i == num_blocks - 1:
                X_lst.append(np.eye(block.num_inputs, dtype=int))
            else:
                X_lst.append(-1 * np.eye(block.num_inputs, dtype=int))
        X = scipy.linalg.block_diag(*X_lst)
        return X

    def generate_ssv_variable(self) -> cvxpy.Variable:
        """Get structured singular value optimization variable for the block structure.

        Returns
        -------
        cvxpy.Variable
            CVXPY variable with specified block structure.
        """
        num_blocks = len(self.block_list)
        X_lst = []
        for i in range(num_blocks):
            row = []
            for j in range(num_blocks):
                # Uncertainty blocks
                block_i = self.block_list[i]
                block_j = self.block_list[j]
                if i == j:
                    # If on the block diagonal, insert variable
                    if (not block_i.is_complex) and (not block_i.is_diagonal):
                        raise NotImplementedError(
                            "Real full perturbations are not supported."
                        )
                    if (not block_i.is_complex) and (block_i.is_diagonal):
                        raise NotImplementedError(
                            "Real diagonal perturbations are not yet supported."
                        )
                    if (block_i.is_complex) and (block_i.is_diagonal):
                        raise NotImplementedError(
                            "Complex diagonal perturbations are not yet supported."
                        )
                    if (block_i.is_complex) and (not block_i.is_square):
                        raise NotImplementedError(
                            "Nonsquare perturbations are not yet supported."
                        )
                    if i == num_blocks - 1:
                        # Last scaling is always identity
                        row.append(np.eye(block_i.num_inputs))
                    else:
                        # Every other scaling is either a scalar or a scalar
                        # multiplied by identity
                        if block_i.num_inputs == 1:
                            xi = cvxpy.Variable((1, 1), complex=True, name=f"x{i}")
                            row.append(xi)
                        else:
                            xi = cvxpy.Variable(1, complex=True, name=f"x{i}")
                            row.append(xi * np.eye(block_i.num_inputs))
                else:
                    # If off the block diagonal, insert zeros
                    row.append(np.zeros((block_i.num_inputs, block_j.num_inputs)))
            X_lst.append(row)
        X = cvxpy.bmat(X_lst)
        return X
