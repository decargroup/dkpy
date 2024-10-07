"""Custom exception classes."""

__all__ = ["TimestepError", "DimensionError", "SolutionError"]


class TimestepError(ValueError):
    """Invalid or incompatible system timesteps."""


class DimensionError(ValueError):
    """Invalid or incompatible array dimensions."""


class SolutionError(RuntimeError):
    """Invalid solver solution."""
