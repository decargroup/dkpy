"""Structured singular value computation."""

__all__ = [
    "StructuredSingularValue",
    "SsvLmiBisection",
]

import abc
from typing import Dict, Tuple, Any, Optional, List

import numpy as np


class StructuredSingularValue(metaclass=abc.ABCMeta):
    """Structured singular value base class."""

    @abc.abstractmethod
    def compute_ssv(
        self,
        N_omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Compute structured singular value.

        Parameters
        ----------
        N_omega : np.ndarray
            Closed-loop transfer function evaluated at each frequency.
        block_structure : np.ndarray
            2D array with 2 columns and as many rows as uncertainty blocks
            in Delta. The columns represent the number of rows and columns in
            each uncertainty block.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
            Structured singular value at each frequency, D-scales at each
            frequency, and solution information. If the structured singular
            value cannot be computed, the first two elements of the tuple are
            ``None``, but solution information is still returned.
        """
        raise NotImplementedError()


class SsvLmiBisection(StructuredSingularValue):
    """Structured singular value using an LMI approach with bisection.

    Synthesis method based on Section 4.25 of [CF24]_.
    """

    def __init__(
        self,
        bisection_atol: float = 1e-5,
        bisection_rtol: float = 1e-4,
        max_iterations: int = 100,
        initial_guess: float = 10,
        lmi_strictness: Optional[float] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = -1,
        objective: str = "constant",
    ):
        """Instantiate :class:`SsvLmiBisection`.

        Solution accuracy depends strongly on the selected solver and
        tolerances. Setting the solver and its tolerances in ``solver_params``
        and setting ``lmi_strictness`` manually is recommended, rather than
        relying on the default settings.

        Parameters
        ----------
        bisection_atol : float
            Bisection absolute tolerance.
        bisection_rtol : float
            Bisection relative tolerance.
        max_iterations : int
            Maximum number of bisection iterations.
        initial_guess : float
            Initial guess for bisection.
        lmi_strictness : Optional[float]
            Strictness for linear matrix inequality constraints. Should be
            larger than the solver tolerance. If ``None``, then it is
            automatically set to 10x the solver's largest absolute tolerance.
        solver_params : Optional[Dict[str, Any]]
            Dictionary of keyword arguments for :func:`cvxpy.Problem.solve`.
            Notable keys are ``'solver'`` and ``'verbose'``. Additional keys
            used to set solver tolerances are solver-dependent. A definitive
            list can be found at [#solvers]_.
        n_jobs : Optional[int]
            Number of processes to use to parallelize the bisection. Set to
            ``None`` for a single thread, or set to ``-1`` (default) to use all
            CPUs. See [#jobs]_.
        objective : str
            Set to ``'constant'`` to solve a feasibility problem at each
            bisection iteration. Set to ``'minimize'`` to minimize the trace of
            the slack variable instead, which may result in better numerical
            conditioning.

        References
        ----------
        .. [#solvers] https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        .. [#jobs] https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib-parallel
        """
        self.bisection_atol = bisection_atol
        self.bisection_rtol = bisection_rtol
        self.max_iterations = max_iterations
        self.initial_guess = initial_guess
        self.lmi_strictness = lmi_strictness
        self.solver_params = solver_params
        self.n_jobs = n_jobs
        self.objective = objective

    def compute_ssv(
        self,
        N_omega: np.ndarray,
        block_structure: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        pass
