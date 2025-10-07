"""Multi-model uncertainty characterization from frequency response data."""

import numpy as np
from matplotlib import pyplot as plt

import dkpy


def example_uncertainty_characterization():
    """Multi-model uncertainty characterization from frequency response data."""

    # Generate example data
    eg = dkpy.example_multimodel_uncertainty()
    complex_response_nom = eg["complex_response_nominal"]
    complex_response_offnom_list = eg["complex_response_offnominal_list"]
    omega = eg["omega"]

    # Plot: Magnitude response of nominal and off-nominal systems
    dkpy.plot_magnitude_response_nom_offnom(
        complex_response_nom, complex_response_offnom_list, omega
    )
    plt.show()

    # Plot: Phase response of nominal and off-nominal systems
    dkpy.plot_phase_response_nom_offnom(
        complex_response_nom, complex_response_offnom_list, omega
    )
    plt.show()

    # Plot: Singular value response of nominal and off-nominal systems
    dkpy.plot_singular_value_response_nom_offnom(
        complex_response_nom, complex_response_offnom_list, omega
    )
    plt.show()

    # Uncertainty models
    uncertainty_models = {"A", "I", "O", "iA", "iI", "iO"}
    complex_response_residuals_dict = dkpy.compute_uncertainty_residual_response(
        complex_response_nom,
        complex_response_offnom_list,
        uncertainty_models,
    )

    # Plot: Singular value response of uncerainty residuals
    dkpy.plot_singular_value_response_uncertainty_residual(
        complex_response_residuals_dict, omega
    )
    plt.show()

    # Plot: Comparison of singular value response of uncerainty residuals for each
    # uncertainty model
    dkpy.plot_singular_value_response_uncertainty_residual_comparison(
        complex_response_residuals_dict, omega
    )
    plt.show()

    # Compute uncertainty weight frequency response
    complex_response_weight_left, complex_response_weight_right = (
        dkpy.compute_optimal_uncertainty_weight_response(
            complex_response_residuals_dict["iA"], "diagonal", "diagonal"
        )
    )

    # Fit overbounding stable and minimum-phase uncertainty weight system
    weight_left = dkpy.fit_overbounding_uncertainty_weight(
        complex_response_weight_left, omega, [4, 5]
    )
    weight_right = dkpy.fit_overbounding_uncertainty_weight(
        complex_response_weight_right, omega, [3, 5]
    )

    # Plot: Magnitude response of uncertainty weight and overbounding uncertainty weight
    # fit
    dkpy.plot_magnitude_response_uncertainty_weight(
        complex_response_weight_left,
        complex_response_weight_right,
        omega,
        weight_left,
        weight_right,
    )
    plt.show()


if __name__ == "__main__":
    example_uncertainty_characterization()
