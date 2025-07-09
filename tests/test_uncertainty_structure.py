"""Test :mod:`uncertainty_structure`."""

import numpy as np
import cvxpy
import pytest

import dkpy


class TestRealDiagonalBlock:
    """Test :class:`RealDiagonalBlock`."""

    def test_real_diagonal_block(self):
        """Test :class:`RealDiagonalBlock`."""
        block = dkpy.RealDiagonalBlock(5)

        assert block.num_inputs == 5
        assert block.num_outputs == 5
        assert block.is_diagonal
        assert not block.is_complex


class TestComplexDiagonalBlock:
    """Test :class:`ComplexDiagonalBlock`."""

    def test_complex_diagonal_block(self):
        """Test :class:`ComplexDiagonalBlock`."""
        block = dkpy.ComplexDiagonalBlock(5)
        assert block.num_inputs == 5
        assert block.num_outputs == 5
        assert block.is_diagonal
        assert block.is_complex


class TestComplexFullBlock:
    """Test :class:`ComplexFullBlock`."""

    def test_complex_full_block(self):
        """Test :class:`ComplexFullBlock`."""
        block = dkpy.ComplexFullBlock(5, 10)
        assert block.num_inputs == 5
        assert block.num_outputs == 10
        assert not block.is_diagonal
        assert block.is_complex


class TestUncertaintyBlockStructure:
    """Test :class:`UncertaintyBlockStructure"""

    @pytest.mark.parametrize(
        "block_structure_matlab, num_inputs_list, num_outputs_list,"
        " is_diagonal_list, is_complex_list",
        [
            (
                np.array([[2, 2], [4, 4], [-3, 0]]),
                [2, 4, 3],
                [2, 4, 3],
                [False, False, True],
                [True, True, False],
            ),
            (
                np.array([[-3, 0], [6, 0], [1, 2]]),
                [3, 6, 2],
                [3, 6, 1],
                [True, True, False],
                [False, True, True],
            ),
            (
                np.array([[3, 6], [-1, 0], [5, 1], [10, 0]]),
                [6, 1, 1, 10],
                [3, 1, 5, 10],
                [False, True, False, True],
                [True, False, True, True],
            ),
            (
                [[2, 2], [4, 4], [-3, 0]],
                [2, 4, 3],
                [2, 4, 3],
                [False, False, True],
                [True, True, False],
            ),
            (
                [[-3, 0], [6, 0], [1, 2]],
                [3, 6, 2],
                [3, 6, 1],
                [True, True, False],
                [False, True, True],
            ),
            (
                [[3, 6], [-1, 0], [5, 1], [10, 0]],
                [6, 1, 1, 10],
                [3, 1, 5, 10],
                [False, True, False, True],
                [True, False, True, True],
            ),
        ],
    )
    def test_convert_block_structure_representation(
        self,
        block_structure_matlab,
        num_inputs_list,
        num_outputs_list,
        is_diagonal_list,
        is_complex_list,
    ):
        uncertainty_structure = dkpy.UncertaintyBlockStructure(block_structure_matlab)

        for idx_block in range(len(uncertainty_structure.block_list)):
            block = uncertainty_structure.block_list[idx_block]
            assert block.num_inputs == num_inputs_list[idx_block]
            assert block.num_outputs == num_outputs_list[idx_block]
            assert block.is_diagonal == is_diagonal_list[idx_block]
            assert block.is_complex == is_complex_list[idx_block]

    @pytest.mark.parametrize(
        "uncertainty_structure, mask_exp",
        [
            (
                dkpy.UncertaintyBlockStructure(
                    [dkpy.ComplexFullBlock(1, 1), dkpy.ComplexFullBlock(1, 1)]
                ),
                np.array(
                    [
                        [-1, 0],
                        [0, 1],
                    ],
                    dtype=int,
                ),
            ),
            (
                dkpy.UncertaintyBlockStructure(
                    [dkpy.ComplexFullBlock(2, 2), dkpy.ComplexFullBlock(1, 1)]
                ),
                np.array(
                    [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                    ],
                    dtype=int,
                ),
            ),
            (
                dkpy.UncertaintyBlockStructure(
                    [dkpy.ComplexFullBlock(1, 1), dkpy.ComplexFullBlock(2, 2)]
                ),
                np.array(
                    [
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=int,
                ),
            ),
        ],
    )
    def test_mask_from_block_structure(self, uncertainty_structure, mask_exp):
        """Test :func:`_mask_from_block_strucure`."""
        mask = uncertainty_structure.generate_d_scale_mask()
        np.testing.assert_allclose(mask_exp, mask)

    @pytest.mark.parametrize(
        "uncertainty_structure, variable_exp",
        [
            (
                dkpy.UncertaintyBlockStructure(
                    [dkpy.ComplexFullBlock(1, 1), dkpy.ComplexFullBlock(2, 2)]
                ),
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
                dkpy.UncertaintyBlockStructure(
                    [
                        dkpy.ComplexFullBlock(1, 1),
                        dkpy.ComplexFullBlock(2, 2),
                        dkpy.ComplexFullBlock(1, 1),
                    ]
                ),
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
    def test_generate_ssv_variable(self, uncertainty_structure, variable_exp):
        """Test :func:`_variable_from_block_structure`."""
        variable = uncertainty_structure.generate_ssv_variable()
        assert variable.ndim == variable_exp.ndim
        assert variable.shape == variable_exp.shape
        # This is the only way I can think of to compare expressions
        assert variable.name() == variable_exp.name()
