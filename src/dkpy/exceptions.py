"""Custom exception classes."""

__all__ = ["TimestepError", "DimensionError", "ConfigError", "SolverError"]


class TimestepError(ValueError):
    """Invalid or incompatible system timesteps."""


class DimensionError(ValueError):
    """Invalid or incompatible array dimensions."""


class ConfigError(ValueError):
    """Incorrect solver or problem settings."""


class SolverError(RuntimeError):
    """Invalid solver solution."""
