import control
import numpy as np
import pytest

import dkpy


class TestComputeUncertaintyResidualResponse:
    """Test :func:`compute_uncertainty_residual_response`."""

    @pytest.mark.parametrize(
        "sys_nom, sys_offnom_list, omega",
        [
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
            ),
        ],
    )
    def test_compute_uncertainty_residual_response(
        self, ndarrays_regression, sys_nom, sys_offnom_list, omega
    ):
        """Regression test :func:`compute_uncertainty_residual_response`."""

        # Frequency response of systems
        frd_nominal = control.frequency_response(sys_nom, omega, squeeze=False)
        frd_offnominal_list = control.frequency_response(
            sys_offnom_list, omega, squeeze=False
        )
        # Complex frequency response of systems
        complex_nominal = frd_nominal.complex.transpose(2, 0, 1)
        complex_offnominal = np.array(
            [frd.complex.transpose(2, 0, 1) for frd in frd_offnominal_list]
        )

        # Uncertainty residual computation
        uncertainty_model_list = [
            "additive",
            "multiplicative_input",
            "multiplicative_output",
            "inverse_additive",
            "inverse_multiplicative_input",
            "inverse_multiplicative_output",
        ]
        frequency_response_residual_dict = dkpy.compute_uncertainty_residual_response(
            complex_nominal,
            complex_offnominal,
            uncertainty_model_list,
        )
        ndarrays_regression.check(
            frequency_response_residual_dict,
            default_tolerance=dict(atol=1e-5, rtol=0),
        )

    @pytest.mark.parametrize(
        "complex_response_nom, complex_response_offnom",
        [
            (
                np.array([[1, 0], [0, 1], [1, 1]]),
                np.array([[7, 8], [9, 10], [11, 12]]),
            ),
        ],
    )
    def test_compute_residual_multiplicative_input_error(
        self, complex_response_nom, complex_response_offnom
    ):
        """Test ValueError of
        :func:`_compute_residual_multiplicative_input`.
        """

        with pytest.raises(ValueError):
            residual_freq = dkpy.uncertainty_characterization._compute_residual_multiplicative_input(
                complex_response_nom, complex_response_offnom
            )

    @pytest.mark.parametrize(
        "complex_response_nom, complex_response_offnom",
        [
            (
                np.array([[1, 0], [0, 1], [1, 1]]).T,
                np.array([[7, 8], [9, 10], [11, 12]]).T,
            ),
        ],
    )
    def test_compute_residual_multiplicative_output_error(
        self, complex_response_nom, complex_response_offnom
    ):
        """Test ValueError of
        :func:`_compute_residual_multiplicative_output`.
        """
        with pytest.raises(ValueError):
            residual_freq = dkpy.uncertainty_characterization._compute_residual_multiplicative_output(
                complex_response_nom, complex_response_offnom
            )

    @pytest.mark.parametrize(
        "complex_response_nom, complex_response_offnom",
        [
            (
                np.array([[1, 0], [0, 1], [1, 1]]),
                np.array([[7, 8], [9, 10], [11, 12]]),
            ),
            (
                np.array([[1, 0], [0, 1], [1, 1]]).T,
                np.array([[7, 8], [9, 10], [11, 12]]).T,
            ),
        ],
    )
    def test_compute_residual_inverse_additive_error(
        self, complex_response_nom, complex_response_offnom
    ):
        """Test ValueError of
        :func:`_compute_residual_inverse_additive`.
        """
        with pytest.raises(ValueError):
            residual_freq = (
                dkpy.uncertainty_characterization._compute_residual_inverse_additive(
                    complex_response_nom, complex_response_offnom
                )
            )

    @pytest.mark.parametrize(
        "complex_response_nom, complex_response_offnom",
        [
            (
                np.array([[1, 0], [0, 1], [1, 1]]),
                np.array([[7, 8], [9, 10], [11, 12]]),
            ),
        ],
    )
    def test_compute_residual_inverse_multiplicative_input_error(
        self, complex_response_nom, complex_response_offnom
    ):
        """Test ValueError of
        :func:`_compute_residual_inverse_multiplicative_input`.
        """
        with pytest.raises(ValueError):
            residual_freq = dkpy.uncertainty_characterization._compute_residual_inverse_multiplicative_input(
                complex_response_nom, complex_response_offnom
            )

    @pytest.mark.parametrize(
        "complex_response_nom, complex_response_offnom",
        [
            (
                np.array([[1, 0], [0, 1], [1, 1]]).T,
                np.array([[7, 8], [9, 10], [11, 12]]).T,
            ),
        ],
    )
    def test_compute_residual_inverse_multiplicative_output_error(
        self, complex_response_nom, complex_response_offnom
    ):
        """Test ValueError of
        :func:`_compute_residual_inverse_multiplicative_output`.
        """
        with pytest.raises(ValueError):
            residual_freq = dkpy.uncertainty_characterization._compute_residual_inverse_multiplicative_output(
                complex_response_nom, complex_response_offnom
            )


class TestComputeUncertaintyWeightResponse:
    """Test :func:`compute_uncertainty_weight_response`."""

    @pytest.mark.parametrize(
        "sys_nom, sys_offnom_list, omega, weight_left_structure, weight_right_structure",
        [
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "diagonal",
                "diagonal",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "diagonal",
                "identity",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "identity",
                "diagonal",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "scalar",
                "identity",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "identity",
                "scalar",
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "diagonal",
                "diagonal",
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "diagonal",
                "identity",
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "identity",
                "diagonal",
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "scalar",
                "identity",
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "identity",
                "scalar",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "diagonal",
                "diagonal",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "diagonal",
                "identity",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "identity",
                "diagonal",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "scalar",
                "identity",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "identity",
                "scalar",
            ),
        ],
    )
    def test_compute_uncertainty_weight_response(
        self,
        ndarrays_regression,
        sys_nom,
        sys_offnom_list,
        omega,
        weight_left_structure,
        weight_right_structure,
    ):
        """Regression test :func:`compute_uncertainty_weight_response`."""
        # Frequency response of systems
        frequency_response_nom = control.frequency_response(
            sys_nom, omega, squeeze=False
        )
        frequency_response_offnom_list = control.frequency_response(
            sys_offnom_list, omega, squeeze=False
        )
        # Complex frequency response of systems
        complex_response_nom = frequency_response_nom.complex.transpose(2, 0, 1)
        complex_response_offnom_list = []
        for frequency_response_offnom in frequency_response_offnom_list:
            complex_response_offnom_list.append(
                frequency_response_offnom.complex.transpose(2, 0, 1)
            )
        complex_response_offnom_list = np.array(
            complex_response_offnom_list, dtype=complex
        )

        # Uncertainty residual computation
        uncertainty_model_list = ["multiplicative_input"]
        complex_response_residual_dict = dkpy.compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            uncertainty_model_list,
        )

        # Optimal uncertainty weight
        complex_response_weight_left, complex_response_weight_right = (
            dkpy.compute_uncertainty_weight_response(
                complex_response_residual_dict["multiplicative_input"],
                weight_left_structure,
                weight_right_structure,
            )
        )
        complex_response_weight_dict = {
            "left": complex_response_weight_left,
            "right": complex_response_weight_right,
        }

        # Regression testing
        ndarrays_regression.check(
            complex_response_weight_dict,
            default_tolerance=dict(atol=1e-3, rtol=0),
        )


class TestTransformUncertaintyModelToAdditive:
    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [(3 * np.eye(3), np.eye(3), np.eye(3), 3 * np.eye(3), np.eye(3), np.eye(3))],
    )
    def test_transform_additive_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import _transform_additive_to_additive

        complex_additive = _transform_additive_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)

    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [
            (
                3 * np.eye(3),
                np.eye(3),
                np.eye(3),
                3 * np.eye(3),
                3 * np.eye(3),
                np.eye(3),
            )
        ],
    )
    def test_transform_multiplicative_input_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import (
            _transform_multiplicative_input_to_additive,
        )

        complex_additive = _transform_multiplicative_input_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)

    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [
            (
                3 * np.eye(3),
                np.eye(3),
                np.eye(3),
                3 * np.eye(3),
                np.eye(3),
                3 * np.eye(3),
            )
        ],
    )
    def test_transform_multiplicative_output_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import (
            _transform_multiplicative_output_to_additive,
        )

        complex_additive = _transform_multiplicative_output_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)

    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [
            (
                3 * np.eye(3),
                0.5 * np.eye(3),
                0.5 * np.eye(3),
                6.85714 * np.eye(3),
                -2.26779 * np.eye(3),
                2.26779 * np.eye(3),
            )
        ],
    )
    def test_transform_inverse_additive_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import (
            _transform_inverse_additive_to_additive,
        )

        complex_additive = _transform_inverse_additive_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)

    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [
            (
                3 * np.eye(3),
                0.5 * np.eye(3),
                0.5 * np.eye(3),
                3.2 * np.eye(3),
                -1.54919 * np.eye(3),
                0.51640 * np.eye(3),
            )
        ],
    )
    def test_transform_inverse_multiplicative_input_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import (
            _transform_inverse_multiplicative_input_to_additive,
        )

        complex_additive = _transform_inverse_multiplicative_input_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)

    @pytest.mark.parametrize(
        "complex_nominal, complex_weight_left, complex_weight_right, "
        "complex_nominal_add_exp, complex_weight_left_add_exp, "
        "complex_weight_right_add_exp",
        [
            (
                3 * np.eye(3),
                0.5 * np.eye(3),
                0.5 * np.eye(3),
                3.2 * np.eye(3),
                -0.51640 * np.eye(3),
                1.54919 * np.eye(3),
            )
        ],
    )
    def test_transform_inverse_multiplicative_output_to_additive(
        self,
        complex_nominal,
        complex_weight_left,
        complex_weight_right,
        complex_nominal_add_exp,
        complex_weight_left_add_exp,
        complex_weight_right_add_exp,
    ):
        from dkpy.uncertainty_characterization import (
            _transform_inverse_multiplicative_output_to_additive,
        )

        complex_additive = _transform_inverse_multiplicative_output_to_additive(
            complex_nominal, complex_weight_left, complex_weight_right
        )
        complex_nominal_add = complex_additive[0]
        complex_weight_left_add = complex_additive[1]
        complex_weight_right_add = complex_additive[2]

        assert np.allclose(complex_nominal_add, complex_nominal_add_exp)
        assert np.allclose(complex_weight_left_add, complex_weight_left_add_exp)
        assert np.allclose(complex_weight_right_add, complex_weight_right_add_exp)


class TestComputeUncertaintyMeasureResponse:
    @pytest.mark.parametrize(
        "nominal, offnominal, omega, uncertainty_model",
        [
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "additive",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "multiplicative_input",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "multiplicative_output",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "inverse_additive",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "inverse_multiplicative_input",
            ),
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "inverse_multiplicative_output",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "additive",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "multiplicative_input",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "multiplicative_output",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "inverse_additive",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "inverse_multiplicative_input",
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "inverse_multiplicative_output",
            ),
        ],
    )
    def test_compute_uncertainty_measure_response(
        self,
        ndarrays_regression,
        nominal,
        offnominal,
        omega,
        uncertainty_model,
    ):
        """Regression test for :func:`compute_uncertainty_measure_response`."""

        # Frequency response of systems
        frd_nominal = control.frequency_response(nominal, omega, squeeze=False)
        frd_offnominal_list = control.frequency_response(
            offnominal, omega, squeeze=False
        )

        # Uncertainty residual computation
        uncertainty_model_list = [uncertainty_model]
        complex_residual_dict = dkpy.compute_uncertainty_residual_response(
            frd_nominal,
            frd_offnominal_list,
            uncertainty_model_list,
        )

        measure_dict = {}
        for weight_structure in ["scalar", "diagonal", "full"]:
            # Uncertainty weight computation
            complex_weight_left, complex_weight_right = (
                dkpy.compute_uncertainty_weight_response(
                    complex_residual_dict[uncertainty_model],
                    weight_structure,
                    weight_structure,
                )
            )

            # Uncertainty measure computation
            measure = dkpy.compute_uncertainty_measure_response(
                frd_nominal,
                complex_weight_left,
                complex_weight_right,
                uncertainty_model,
            )

            measure_dict[weight_structure] = measure

        ndarrays_regression.check(
            measure_dict,
            default_tolerance=dict(atol=1e-2, rtol=1e-2),
        )


class TestFitUncertaintyWeight:
    """Test :func:`fit_uncertainty_weight`."""

    @pytest.mark.parametrize(
        "sys_nom, sys_offnom_list, omega, weight_left_structure, weight_right_structure, fit_order",
        [
            (
                control.TransferFunction([1], [0.5, 1]),
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
                "diagonal",
                "diagonal",
                4,
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
                "diagonal",
                "diagonal",
                4,
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "diagonal",
                "diagonal",
                4,
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
                "diagonal",
                "diagonal",
                [4, 4],
            ),
        ],
    )
    def test_fit_uncertainty_weight(
        self,
        ndarrays_regression,
        sys_nom,
        sys_offnom_list,
        omega,
        weight_left_structure,
        weight_right_structure,
        fit_order,
    ):
        """Regression test :func:`fit_uncertainty_weight`."""
        # Frequency response of systems
        frequency_response_nom = control.frequency_response(
            sys_nom, omega, squeeze=False
        )
        frequency_response_offnom_list = control.frequency_response(
            sys_offnom_list, omega, squeeze=False
        )
        # Complex frequency response of systems
        complex_response_nom = frequency_response_nom.complex.transpose(2, 0, 1)
        complex_response_offnom_list = []
        for frequency_response_offnom in frequency_response_offnom_list:
            complex_response_offnom_list.append(
                frequency_response_offnom.complex.transpose(2, 0, 1)
            )
        complex_response_offnom_list = np.array(
            complex_response_offnom_list, dtype=complex
        )

        # Uncertainty residual computation
        uncertainty_model_list = ["multiplicative_input"]
        complex_response_residual_dict = dkpy.compute_uncertainty_residual_response(
            complex_response_nom,
            complex_response_offnom_list,
            uncertainty_model_list,
        )

        # Optimal uncertainty weight
        complex_response_weight_left, complex_response_weight_right = (
            dkpy.compute_uncertainty_weight_response(
                complex_response_residual_dict["multiplicative_input"],
                weight_left_structure,
                weight_right_structure,
            )
        )

        # Overbounding transfer function fit
        weight_left_fit = dkpy.fit_uncertainty_weight(
            complex_response_weight_left,
            omega,
            fit_order,
            "left",
            weight_left_structure,
        )
        weight_right_fit = dkpy.fit_uncertainty_weight(
            complex_response_weight_right,
            omega,
            fit_order,
            "right",
            weight_right_structure,
        )

        # Uncertainty weight fit frequency response
        frequency_response_weight_left_fit = control.frequency_response(
            weight_left_fit, omega, squeeze=False
        )
        frequency_response_weight_right_fit = control.frequency_response(
            weight_right_fit, omega, squeeze=False
        )
        complex_response_weight_left_fit = (
            frequency_response_weight_left_fit.complex.transpose(2, 0, 1)
        )
        complex_response_weight_right_fit = (
            frequency_response_weight_right_fit.complex.transpose(2, 0, 1)
        )
        complex_response_weight_fit_dict = {
            "left": complex_response_weight_left_fit,
            "right": complex_response_weight_right_fit,
        }

        # Regression testing
        ndarrays_regression.check(
            complex_response_weight_fit_dict,
            default_tolerance=dict(atol=1e-2, rtol=1e-2),
        )


class TestConvertFrequencyResponseDataToArray:
    @pytest.mark.parametrize(
        "sys, omega",
        [
            (
                control.TransferFunction([1], [0.5, 1]),
                np.logspace(-2, 2, 100),
            ),
            (
                control.TransferFunction([1], [1, 2 * 0.5 * 1, 1**2]),
                np.logspace(-1, 1, 100),
            ),
            (
                control.TransferFunction(
                    [
                        [[1], [3]],
                        [[2], [1]],
                    ],
                    [
                        [[1, 2 * 0.5 * 1, 1**2], [0.5, 1]],
                        [[1, 1], [1, 2 * 0.3 * 5, 5**2]],
                    ],
                ),
                np.logspace(-1, 1.5, 100),
            ),
        ],
    )
    def test_convert_frequency_response_data_to_array(self, sys, omega):
        # Desired frequency response array shape
        complex_response_shape_true = (omega.size, sys.noutputs, sys.ninputs)

        # Evaluate frequency response
        frequency_response = control.frequency_response(sys, omega)

        # Convert frequency response data
        complex_response = (
            dkpy.uncertainty_characterization._convert_frequency_response_data_to_array(
                frequency_response
            )
        )

        # Verify type
        assert isinstance(complex_response, np.ndarray)
        # Verify dimensions
        assert complex_response.shape == complex_response_shape_true
        # Verify complex datatype
        assert np.all(np.iscomplex(complex_response))
        # Verify that the function does not affect responses in the correct format
        assert np.all(
            np.isclose(
                complex_response,
                dkpy.uncertainty_characterization._convert_frequency_response_data_to_array(
                    complex_response
                ),
            ),
        )

    @pytest.mark.parametrize(
        "complex_response",
        [
            np.ones(100, dtype=complex),
            np.ones((100, 5), dtype=complex),
            np.ones((100, 5, 5, 5), dtype=complex),
        ],
    )
    def test_convert_frequency_response_data_to_array_error(self, complex_response):
        with pytest.raises(ValueError):
            complex_response = dkpy.uncertainty_characterization._convert_frequency_response_data_to_array(
                complex_response
            )


class TestConvertFrequencyResponseListToArray:
    @pytest.mark.parametrize(
        "sys_list, omega",
        [
            (
                [
                    control.TransferFunction([1], [0.3, 1]),
                    control.TransferFunction([1], [0.4, 1]),
                    control.TransferFunction([1], [0.6, 1]),
                    control.TransferFunction([1], [0.7, 1]),
                ],
                np.logspace(-2, 2, 100),
            ),
            (
                [
                    control.TransferFunction([1], [1, 2 * 0.3 * 1.5, 1.5**2]),
                    control.TransferFunction([1], [1, 2 * 0.7 * 1.3, 1.3**2]),
                    control.TransferFunction([1], [1, 2 * 0.2 * 0.9, 0.9**2]),
                    control.TransferFunction([1], [1, 2 * 0.9 * 0.7, 0.7**2]),
                ],
                np.logspace(-1, 1, 100),
            ),
            (
                [
                    control.TransferFunction(
                        [
                            [[1], [3.25]],
                            [[1.9], [1]],
                        ],
                        [
                            [[1, 2 * 0.4 * 1.1, 1.1**2], [0.6, 1]],
                            [[0.8, 1], [1, 2 * 0.4 * 4.2, 4.2**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1.1], [2.85]],
                            [[1.7], [0.9]],
                        ],
                        [
                            [[1, 2 * 0.45 * 0.9, 0.9**2], [0.55, 1]],
                            [[0.95, 1], [1, 2 * 0.8 * 5.5, 5.5**2]],
                        ],
                    ),
                    control.TransferFunction(
                        [
                            [[1], [3.05]],
                            [[1.7], [1.0]],
                        ],
                        [
                            [[1, 2 * 0.5 * 1.3, 1.3**2], [0.42, 1]],
                            [[1, 1], [1, 2 * 0.9 * 5.25, 5.25**2]],
                        ],
                    ),
                ],
                np.logspace(-1, 1.5, 100),
            ),
        ],
    )
    def test_convert_frequency_response_list_to_array(self, sys_list, omega):
        # Desired frequency response array shape
        complex_response_list_shape_true = (
            len(sys_list),
            omega.size,
            sys_list[0].noutputs,
            sys_list[0].ninputs,
        )

        # Evaluate frequency response
        frequency_response_list = control.frequency_response(sys_list, omega)

        # Convert frequency response data
        complex_response_list = (
            dkpy.uncertainty_characterization._convert_frequency_response_list_to_array(
                frequency_response_list
            )
        )

        # Verify type
        assert isinstance(complex_response_list, np.ndarray)
        # Verify dimensions
        assert complex_response_list.shape == complex_response_list_shape_true
        # Verify complex datatype
        assert np.all(np.iscomplex(complex_response_list))
        # Verify that the function does not affect responses in the correct format
        assert np.all(
            np.isclose(
                complex_response_list,
                dkpy.uncertainty_characterization._convert_frequency_response_list_to_array(
                    complex_response_list
                ),
            ),
        )

    @pytest.mark.parametrize(
        "complex_response_list",
        [
            np.ones(100, dtype=complex),
            np.ones((100, 5), dtype=complex),
            np.ones((100, 5, 5), dtype=complex),
            np.ones((100, 5, 5, 5, 5), dtype=complex),
        ],
    )
    def test_convert_frequency_response_list_to_array_error(
        self, complex_response_list
    ):
        with pytest.raises(ValueError):
            complex_response_list = dkpy.uncertainty_characterization._convert_frequency_response_list_to_array(
                complex_response_list
            )
