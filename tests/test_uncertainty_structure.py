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
        "block_structure, num_inputs_list, num_outputs_list,"
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
            (
                [
                    dkpy.ComplexFullBlock(2, 2),
                    dkpy.ComplexFullBlock(4, 4),
                    dkpy.RealDiagonalBlock(3),
                ],
                [2, 4, 3],
                [2, 4, 3],
                [False, False, True],
                [True, True, False],
            ),
            (
                [
                    dkpy.RealDiagonalBlock(3),
                    dkpy.ComplexDiagonalBlock(6),
                    dkpy.ComplexFullBlock(2, 1),
                ],
                [3, 6, 2],
                [3, 6, 1],
                [True, True, False],
                [False, True, True],
            ),
            (
                [
                    dkpy.ComplexFullBlock(6, 3),
                    dkpy.RealDiagonalBlock(1),
                    dkpy.ComplexFullBlock(1, 5),
                    dkpy.ComplexDiagonalBlock(10),
                ],
                [6, 1, 1, 10],
                [3, 1, 5, 10],
                [False, True, False, True],
                [True, False, True, True],
            ),
        ],
    )
    def test_convert_block_structure_representation(
        self,
        block_structure,
        num_inputs_list,
        num_outputs_list,
        is_diagonal_list,
        is_complex_list,
    ):
        block_structure = (
            dkpy.uncertainty_structure._convert_block_structure_representation(
                block_structure
            )
        )

        for idx_block in range(len(block_structure)):
            block = block_structure[idx_block]
            assert block.num_inputs == num_inputs_list[idx_block]
            assert block.num_outputs == num_outputs_list[idx_block]
            assert block.is_diagonal == is_diagonal_list[idx_block]
            assert block.is_complex == is_complex_list[idx_block]
