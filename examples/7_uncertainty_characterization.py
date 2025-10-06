"""Multi-model uncertainty characterization from frequency response data."""

import numpy as np
from matplotlib import pyplot as plt

import dkpy


def example_uncertainty_characterization():
    """Multi-model uncertainty characterization from frequency response data."""

    # Generate example data
    eg = dkpy.example_multimodel_uncertainty()
    frequency_response_nom = eg["frequency_response_nominal"]
    frequency_response_offnom_list = eg["frequency_response_offnominal_list"]

    # Plot: Magnitude response of nominal and off-nominal systems
    dkpy.plot_magnitude_response_nom_offnom(
        frequency_response_nom, frequency_response_offnom_list
    )
    plt.show()

    # Plot: Phase response of nominal and off-nominal systems
    dkpy.plot_phase_response_nom_offnom(
        frequency_response_nom, frequency_response_offnom_list
    )
    plt.show()

    # Plot: Singular value response of nominal and off-nominal systems
    dkpy.plot_singular_value_response_nom_offnom(
        frequency_response_nom, frequency_response_offnom_list
    )
    plt.show()

    # Uncertainty models
    uncertainty_models = {"A", "I", "O", "iA", "iI", "iO"}
    uncertainty_residuals_dict = dkpy.compute_uncertainty_residual_response(
        frequency_response_nom,
        frequency_response_offnom_list,
        uncertainty_models,
    )

    # Plot: Singular value response of uncerainty residuals
    dkpy.plot_singular_value_response_uncertainty_residual(uncertainty_residuals_dict)
    plt.show()

    # Plot: Comparison of singular value response of uncerainty residuals for each
    # uncertainty model
    dkpy.plot_singular_value_response_uncertainty_residual_comparison(
        uncertainty_residuals_dict
    )
    plt.show()

    # Compute uncertainty weight frequency response
    response_weight_left, response_weight_right = (
        dkpy.compute_optimal_uncertainty_weight_response(
            uncertainty_residuals_dict["iA"], "diagonal", "diagonal"
        )
    )

    # Fit overbounding stable and minimum-phase uncertainty weight system
    weight_left = dkpy.fit_overbounding_uncertainty_weight(response_weight_left, [4, 5])
    weight_right = dkpy.fit_overbounding_uncertainty_weight(
        response_weight_right, [3, 5]
    )

    # Plot: Magnitude response of uncertainty weight and overbounding uncertainty weight
    # fit
    dkpy.plot_magnitude_response_uncertainty_weight(
        response_weight_left, response_weight_right, weight_left, weight_right
    )
    plt.show()


if __name__ == "__main__":
    example_uncertainty_characterization()
