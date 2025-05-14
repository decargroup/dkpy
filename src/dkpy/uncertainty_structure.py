"""Uncertainty block."""


class RealDiagonalBlock:
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
        self.num_inputs = num_channels
        self.num_outputs = num_channels


class ComplexDiagonalBlock:
    """Complex-valued diagonal uncertainty block."""

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
        self.num_inputs = num_channels
        self.num_outputs = num_channels


class ComplexFullBlock:
    """Complex-valued full uncertainty block."""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Instantiate :class:`RealDiagonalBlock`.

        Parameters
        ----------
        num_channels : int
            Number of inputs/outputs to the uncertainty block.
        """
        # Error handling
        if num_inputs <= 0:
            raise ValueError("num_inputs must be greater than 0.")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be greater than 0.")

        # Uncertainty block parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
