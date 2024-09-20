"""Controller synthesis classes."""

import abc
from typing import Any, Dict, Tuple

import control


class ControllerSynthesis(metaclass=abc.ABCMeta):
    """Controller synthesis base class."""

    @abc.abstractclassmethod
    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        """Synthesize controller.

        Parameters
        ----------
        P : control.StateSpace
            Generalized plant, with ``y`` and ``u`` as last outputs and inputs
            respectively.
        n_y : int
            Number of measurements (controller inputs).
        n_u : int
            Number of controller outputs.

        Returns
        -------
        Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]
            Controller, closed-loop system, objective function value, solution
            information.
        """
        raise NotImplementedError()


class HinfSynSlicot(ControllerSynthesis):
    """H-infinity synthesis using SLICOT's Riccati equation method."""

    def __init__(self):
        """Instantiate :class:`HinfSynSlicot`."""
        pass

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        K, N, gamma, rcond = control.hinfsyn(P, n_y, n_u)
        info = {
            "rcond": rcond,
        }
        return K, N, gamma, info


class HinfSynLmi(ControllerSynthesis):
    """H-infinity synthesis using a linear matrix inequality approach."""

    def __init__(self):
        """Instantiate :class:`HinfSynLmi`."""
        raise NotImplementedError()  # TODO

    def synthesize(
        self,
        P: control.StateSpace,
        n_y: int,
        n_u: int,
    ) -> Tuple[control.StateSpace, control.StateSpace, float, Dict[str, Any]]:
        raise NotImplementedError()  # TODO
