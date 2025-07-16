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
                cvxpy.bmat(
                    [
                        [
                            cvxpy.Variable((1, 1), complex=True, name="x0"),
                            np.zeros((1, 2)),
                        ],
                        [
                            np.zeros((2, 1)),
                            np.eye(2),
                        ],
                    ]
                ),
            ),
            (
                [
                    dkpy.ComplexFullBlock(1, 1),
                    dkpy.ComplexFullBlock(2, 2),
                    dkpy.ComplexFullBlock(1, 1),
                ],
                cvxpy.bmat(
                    [
                        [
                            cvxpy.Variable((1, 1), complex=True, name="x0"),
                            np.zeros((1, 2)),
                            np.zeros((1, 1)),
                        ],
                        [
                            np.zeros((2, 1)),
                            cvxpy.Variable(1, complex=True, name="x1") * np.eye(2),
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
        ],
    )
    def test_generate_ssv_variable(self, block_structure, variable_exp):
        """Test :func:`_variable_from_block_structure`."""
        variable = dkpy.structured_singular_value._generate_ssv_variable(
            block_structure
        )
        assert variable.ndim == variable_exp.ndim
        assert variable.shape == variable_exp.shape
        # This is the only way I can think of to compare expressions
        assert variable.name() == variable_exp.name()
