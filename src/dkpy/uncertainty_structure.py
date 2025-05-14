"""Uncertainty block."""


class UncertaintyBlock:
    """Generic uncertainty block."""

    def __init__(
        self, num_inputs: int, num_outputs: int, is_diagonal: bool, is_complex: bool
    ):
        """Instantiate :class:`UncertaintyBlock`.

        Parameters
        ----------
        num_inputs : int
            Number of inputs of the uncertainty block.
        num_outputs : int
            Number of outputs of the uncertainty block.
        """
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
        """
        # Error handling
        if num_channels <= 0:
            raise ValueError("num_channels must be greater than 0.")

        # Uncertainty block parameters
        self._num_inputs = num_channels
        self._num_outputs = num_channels
        self._is_diagonal = True
        self._is_complex = False


class ComplexDiagonalBlock(UncertaintyBlock):
    """Complex-valued diagonal uncertainty block."""

    def __init__(self, num_channels: int):
        """Instantiate :class:`ComplexDiagonalBlock`.

        Parameters
        ----------
        num_channels : int
            Number of inputs/outputs to the uncertainty block.
        """
        # Error handling
        if num_channels <= 0:
            raise ValueError("num_channels must be greater than 0.")

        # Uncertainty block parameters
        self._num_inputs = num_channels
        self._num_outputs = num_channels
        self._is_diagonal = True
        self._is_complex = True


class ComplexFullBlock(UncertaintyBlock):
    """Complex-valued full uncertainty block."""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Instantiate :class:`RealDiagonalBlock`.

        Parameters
        ----------
        num_inputs : int
            Number of inputs to the uncertainty block.
        num_outputs : int
            Number of inputs to the uncertainty block.
        """
        # Error handling
        if num_inputs <= 0:
            raise ValueError("num_inputs must be greater than 0.")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be greater than 0.")

        # Uncertainty block parameters
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._is_diagonal = False
        self._is_complex = True
