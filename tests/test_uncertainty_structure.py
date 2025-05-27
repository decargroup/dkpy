"""Test :mod:`uncertainty_structure`."""

import numpy as np
import pytest

import dkpy


class TestRealDiagonalBlock:
    """Test :func:`RealDiagonalBlock`."""

    def test_real_diagonal_block(self):
        """Test :func:`RealDiagonalBlock`."""
        block = dkpy.RealDiagonalBlock(5)

        assert block.num_inputs == 5
        assert block.num_outputs == 5
        assert block.is_diagonal
        assert not block.is_complex


class TestComplexDiagonalBlock:
    """Test :func:`ComplexDiagonalBlock`."""

    def test_real_diagonal_block(self):
        """Test :func:`ComplexDiagonalBlock`."""
        block = dkpy.ComplexDiagonalBlock(5)

        assert block.num_inputs == 5
        assert block.num_outputs == 5
        assert block.is_diagonal
        assert block.is_complex


class TestComplexFullBlock:
    """Test :func:`ComplexFullBlock`."""

    def test_real_diagonal_block(self):
        """Test :func:`ComplexFullBlock`."""
        block = dkpy.ComplexFullBlock(5, 10)

        assert block.num_inputs == 5
        assert block.num_outputs == 10
        assert not block.is_diagonal
        assert block.is_complex


class TestConvertMatlabBlockStructure:
    """Test :func:`_convert_matlab_block_structure`."""

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
        ],
    )
    def test_convert_matlab_block_structure(
        self,
        block_structure_matlab,
        num_inputs_list,
        num_outputs_list,
        is_diagonal_list,
        is_complex_list,
    ):
        """Test :func:`_convert_matlab_block_structure`."""
        block_structure = dkpy.uncertainty_structure._convert_matlab_block_structure(
            block_structure_matlab
        )

        for idx_block in range(len(block_structure)):
            block = block_structure[idx_block]
            assert block.num_inputs == num_inputs_list[idx_block]
            assert block.num_outputs == num_outputs_list[idx_block]
            assert block.is_diagonal == is_diagonal_list[idx_block]
            assert block.is_complex == is_complex_list[idx_block]
