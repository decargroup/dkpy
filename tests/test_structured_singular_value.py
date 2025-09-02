"""Test :mod:`structured_singular_value`."""

import cvxpy
import numpy as np
import pytest

import dkpy


class TestGenerateSsvVariable:
    @pytest.mark.parametrize(
        "block_structure, variable_exp",
        [
            (
                [dkpy.ComplexFullBlock(1, 1), dkpy.ComplexFullBlock(2, 2)],
                (
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((), complex=True, name="x0") * np.eye(1),
                                np.zeros((1, 2)),
                            ],
                            [
                                np.zeros((2, 1)),
                                np.eye(2),
                            ],
                        ]
                    ),
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((), complex=True, name="x0") * np.eye(1),
                                np.zeros((1, 2)),
                            ],
                            [
                                np.zeros((2, 1)),
                                np.eye(2),
                            ],
                        ]
                    ),
                ),
            ),
            (
                [
                    dkpy.ComplexFullBlock(1, 1),
                    dkpy.ComplexFullBlock(2, 2),
                    dkpy.ComplexFullBlock(1, 1),
                ],
                (
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((), complex=True, name="x0") * np.eye(1),
                                np.zeros((1, 2)),
                                np.zeros((1, 1)),
                            ],
                            [
                                np.zeros((2, 1)),
                                cvxpy.Variable((), complex=True, name="x1") * np.eye(2),
                                np.zeros((2, 1)),
                            ],
                            [
                                np.zeros((1, 1)),
                                np.zeros((1, 2)),
                                np.eye(1),
                            ],
                        ]
                    ),
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((), complex=True, name="x0") * np.eye(1),
                                np.zeros((1, 2)),
                                np.zeros((1, 1)),
                            ],
                            [
                                np.zeros((2, 1)),
                                cvxpy.Variable((), complex=True, name="x1") * np.eye(2),
                                np.zeros((2, 1)),
                            ],
                            [
                                np.zeros((1, 1)),
                                np.zeros((1, 2)),
                                np.eye(1),
                            ],
                        ]
                    ),
                ),
            ),
            (
                [
                    dkpy.ComplexDiagonalBlock(2),
                    dkpy.ComplexDiagonalBlock(1),
                    dkpy.ComplexFullBlock(3, 2),
                    dkpy.ComplexFullBlock(3, 3),
                    dkpy.ComplexDiagonalBlock(3),
                    dkpy.ComplexFullBlock(1, 2),
                ],
                (
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((2, 2), hermitian=True, name="X0"),
                                np.zeros((2, 1)),
                                np.zeros((2, 3)),
                                np.zeros((2, 3)),
                                np.zeros((2, 3)),
                                np.zeros((2, 1)),
                            ],
                            [
                                np.zeros((1, 2)),
                                cvxpy.Variable((1, 1), hermitian=True, name="X1"),
                                np.zeros((1, 3)),
                                np.zeros((1, 3)),
                                np.zeros((1, 3)),
                                np.zeros((1, 1)),
                            ],
                            [
                                np.zeros((3, 2)),
                                np.zeros((3, 1)),
                                cvxpy.Variable((), complex=True, name="x2") * np.eye(3),
                                np.zeros((3, 3)),
                                np.zeros((3, 3)),
                                np.zeros((3, 1)),
                            ],
                            [
                                np.zeros((3, 2)),
                                np.zeros((3, 1)),
                                np.zeros((3, 3)),
                                cvxpy.Variable((), complex=True, name="x3") * np.eye(3),
                                np.zeros((3, 3)),
                                np.zeros((3, 1)),
                            ],
                            [
                                np.zeros((3, 2)),
                                np.zeros((3, 1)),
                                np.zeros((3, 3)),
                                np.zeros((3, 3)),
                                cvxpy.Variable((3, 3), hermitian=True, name="X4"),
                                np.zeros((3, 1)),
                            ],
                            [
                                np.zeros((1, 2)),
                                np.zeros((1, 1)),
                                np.zeros((1, 3)),
                                np.zeros((1, 3)),
                                np.zeros((1, 3)),
                                np.eye(1),
                            ],
                        ],
                    ),
                    cvxpy.bmat(
                        [
                            [
                                cvxpy.Variable((2, 2), hermitian=True, name="X0"),
                                np.zeros((2, 1)),
                                np.zeros((2, 2)),
                                np.zeros((2, 3)),
                                np.zeros((2, 3)),
                                np.zeros((2, 2)),
                            ],
                            [
                                np.zeros((1, 2)),
                                cvxpy.Variable((1, 1), hermitian=True, name="X1"),
                                np.zeros((1, 2)),
                                np.zeros((1, 3)),
                                np.zeros((1, 3)),
                                np.zeros((1, 2)),
                            ],
                            [
                                np.zeros((2, 2)),
                                np.zeros((2, 1)),
                                cvxpy.Variable((), name="x2") * np.eye(2),
                                np.zeros((2, 3)),
                                np.zeros((2, 3)),
                                np.zeros((2, 2)),
                            ],
                            [
                                np.zeros((3, 2)),
                                np.zeros((3, 1)),
                                np.zeros((3, 2)),
                                cvxpy.Variable((), complex=True, name="x3") * np.eye(3),
                                np.zeros((3, 3)),
                                np.zeros((3, 2)),
                            ],
                            [
                                np.zeros((3, 2)),
                                np.zeros((3, 1)),
                                np.zeros((3, 2)),
                                np.zeros((3, 3)),
                                cvxpy.Variable((3, 3), hermitian=True, name="X4"),
                                np.zeros((3, 2)),
                            ],
                            [
                                np.zeros((2, 2)),
                                np.zeros((2, 1)),
                                np.zeros((2, 2)),
                                np.zeros((2, 3)),
                                np.zeros((2, 3)),
                                np.eye(2),
                            ],
                        ],
                    ),
                ),
            ),
        ],
    )
    def test_generate_ssv_variable(self, block_structure, variable_exp):
        """Test :func:`_variable_from_block_structure`."""
        variable = dkpy.structured_singular_value._generate_ssv_variable(
            block_structure
        )
        assert variable[0].ndim == variable_exp[0].ndim
        assert variable[1].ndim == variable_exp[1].ndim
        assert variable[0].shape == variable_exp[0].shape
        assert variable[1].shape == variable_exp[1].shape
        assert variable[0].name() == variable_exp[0].name()
        assert variable[1].name() == variable_exp[1].name()


class TestSsvLmiBisection:
    @pytest.mark.parametrize(
        "N_omega, block_structure, mu_exp",
        [
            (
                np.array(
                    [
                        [1j, 2, 2j, 0, -1, -1 + 3j, 2 + 3j],
                        [3 + 1j, 2 - 1j, -1 + 1j, 2 + 1j, -1 + 1j, 1, -1 + 1j],
                        [3 + 1j, 1j, 2 + 2j, -1 + 2j, 3 - 1j, 3j, -1 + 1j],
                        [-1 + 1j, -1 - 1j, 1j, 0, 1 - 1j, 2 - 1j, 2 + 2j],
                        [3, 1j, 1 + 1j, 3j, 1 + 1j, 3j, -1j],
                        [1, 3 + 2j, 2 + 2j, 3j, 1 + 2j, 2 + 1j, -1 + 2j],
                        [2 + 1j, -1 - 1j, -1, 3 + 3j, 2 + 3j, 2j, 1 - 1j],
                    ]
                ),
                [
                    dkpy.ComplexDiagonalBlock(2),
                    dkpy.ComplexDiagonalBlock(1),
                    dkpy.ComplexFullBlock(3, 2),
                    dkpy.ComplexFullBlock(1, 2),
                ],
                10.5523,
            ),
            (
                np.array(
                    [
                        [1j, 2, 2j, 0],
                        [3 + 1j, 2 - 1j, -1 + 1j, 2 + 1j],
                        [3 + 1j, 1j, 2 + 2j, -1 + 2j],
                        [3 - 1j, 3j, -1 + 1j, 2 + 2j],
                        [-1 + 1j, -1 - 1j, 1j, 0],
                        [1 - 1j, 2 - 1j, 2 + 2j, 1],
                        [3, 1j, 1 + 1j, 3j],
                        [1 + 2j, 2 + 1j, -1 + 2j, 1j],
                        [2 + 1j, -1 - 1j, -1, 3 + 3j],
                        [2 + 3j, 2j, 1 - 1j, 0],
                    ]
                ),
                [
                    dkpy.ComplexFullBlock(4, 2),
                    dkpy.ComplexFullBlock(6, 2),
                ],
                10.2438,
            ),
        ],
    )
    def test_compute_ssv(self, N_omega, block_structure, mu_exp):
        N_omega = np.expand_dims(N_omega, N_omega.ndim)
        mu, D_l_scales, D_r_scales, info = dkpy.SsvLmiBisection(n_jobs=1).compute_ssv(
            N_omega, block_structure
        )
        assert np.isclose(mu, mu_exp)
