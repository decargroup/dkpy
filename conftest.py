"""Pytest fixtures for doctests."""

import control
import numpy as np
import pytest
import scipy.linalg

import dkpy


def _tf_mat_mul(tf, mat):
    """TODO Remove."""
    out = np.zeros(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out[i, j] = mat[i, j] * tf
    out_comb = _combine(out)
    return out_comb


def _combine(G):
    """TODO Remove."""
    G = np.array(G)
    num = []
    den = []
    for i_out in range(G.shape[0]):
        for j_out in range(G[i_out, 0].noutputs):
            num_row = []
            den_row = []
            for i_in in range(G.shape[1]):
                for j_in in range(G[i_out, i_in].ninputs):
                    num_row.append(G[i_out, i_in].num[j_out][j_in])
                    den_row.append(G[i_out, i_in].den[j_out][j_in])
            num.append(num_row)
            den.append(den_row)
    G_tf = control.TransferFunction(num, den, dt=G[0][0].dt)
    return G_tf


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
    G = _tf_mat_mul(control.TransferFunction([1], [75, 1]), G0)
    # Weights
    Wp = _tf_mat_mul(
        control.TransferFunction([10, 1], [10, 1e-5]),
        0.5 * np.eye(2),
    )
    Wi = _tf_mat_mul(
        control.TransferFunction([1, 0.2], [0.5, 1]),
        np.eye(2),
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
    K = _tf_mat_mul(
        control.TransferFunction([75, 1], [1, 1e-5]),
        0.7 * scipy.linalg.inv(G0),
    )
    doctest_namespace["example_skogestad2006"] = P, n_y, n_u, K
