"""Pytest fixtures for doctests."""

import control
import numpy as np
import pytest
import scipy.linalg

import dkpy


@pytest.fixture(autouse=True)
def add_dkpy(doctest_namespace):
    """Add ``dkpy`` to namespace."""
    doctest_namespace["dkpy"] = dkpy


@pytest.fixture(autouse=True)
def add_example_scherer1997(doctest_namespace):
    """Add generalized plant from [SGC97]_, Example 7."""
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
    doctest_namespace["example_scherer1997"] = P, n_y, n_u


@pytest.fixture(autouse=True)
def add_example_skogestad2006(doctest_namespace):
    """Add generalized plant from [SP06]_, Table 8.1."""
    # Plant
    G0 = np.array(
        [
            [87.8, -86.4],
            [108.2, -109.6],
        ]
    )
    G = control.append(
        control.TransferFunction([1], [75, 1]),
        control.TransferFunction([1], [75, 1]),
    ) * control.TransferFunction(
        G0.reshape(2, 2, 1),
        np.ones((2, 2, 1)),
    )
    # Weights
    Wp = 0.5 * control.append(
        control.TransferFunction([10, 1], [10, 1e-5]),
        control.TransferFunction([10, 1], [10, 1e-5]),
    )
    Wi = control.append(
        control.TransferFunction([1, 0.2], [0.5, 1]),
        control.TransferFunction([1, 0.2], [0.5, 1]),
    )
    G.name = "G"
    Wp.name = "Wp"
    Wi.name = "Wi"
    sum_w = control.summing_junction(
        inputs=["u_w", "u_G"],
        dimension=2,
        name="sum_w",
    )
    sum_del = control.summing_junction(
        inputs=["u_del", "u_u"],
        dimension=2,
        name="sum_del",
    )
    split = control.summing_junction(
        inputs=["u"],
        dimension=2,
        name="split",
    )
    P = control.interconnect(
        syslist=[G, Wp, Wi, sum_w, sum_del, split],
        connections=[
            ["G.u", "sum_del.y"],
            ["sum_del.u_u", "split.y"],
            ["sum_w.u_G", "G.y"],
            ["Wp.u", "sum_w.y"],
            ["Wi.u", "split.y"],
        ],
        inplist=["sum_del.u_del", "sum_w.u_w", "split.u"],
        outlist=["Wi.y", "Wp.y", "-sum_w.y"],
    )
    # Dimensions
    n_y = 2
    n_u = 2
    # Inverse-based controller
    K = (
        0.7
        * control.append(
            control.TransferFunction([75, 1], [1, 1e-5]),
            control.TransferFunction([75, 1], [1, 1e-5]),
        )
        * control.TransferFunction(
            scipy.linalg.inv(G0).reshape(2, 2, 1),
            np.ones((2, 2, 1)),
        )
    )
    doctest_namespace["example_skogestad2006"] = P, n_y, n_u, K
