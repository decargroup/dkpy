"""Pytest fixtures for doctests."""

import control
import numpy as np
import pytest

import dkpy


@pytest.fixture(autouse=True)
def add_dkpy(doctest_namespace):
    """Add ``dkpy`` to namespace."""
    doctest_namespace["dkpy"] = dkpy


@pytest.fixture(autouse=True)
def add_generalized_plant(doctest_namespace):
    """Add generalized plant to namespace.

    Based on Example 7 from [SGC97]_.
    """
    # Process model
    A = np.array([[0, 10, 2], [-1, 1, 0], [0, 2, -5]])
    B1 = np.array([[1], [0], [1]])
    B2 = np.array([[0], [1], [0]])
    # Plant output
    C2 = np.array([[0, 1, 0]])
    D21 = np.array([[2]])
    D22 = np.array([[0]])
    # Hinf performance
    C1 = np.array([[1, 0, 0], [0, 0, 0]])
    D11 = np.array([[0], [0]])
    D12 = np.array([[0], [1]])
    # Dimensions
    n_y = 1
    n_u = 1
    # Create generalized plant
    B_gp = np.block([B1, B2])
    C_gp = np.block([[C1], [C2]])
    D_gp = np.block([[D11, D12], [D21, D22]])
    P = control.StateSpace(A, B_gp, C_gp, D_gp)
    doctest_namespace["P"] = P
    doctest_namespace["n_y"] = n_y
    doctest_namespace["n_u"] = n_u
