"""Multi-model uncertainty characterization from frequency response data."""

from matplotlib import pyplot as plt

import dkpy


def example_uncertainty_characterization():
    """Multi-model uncertainty characterization from frequency response data."""

    # Generate example data
    eg = dkpy.example_multimodel_uncertainty()
    response_nom = eg["complex_response_nominal"]
    response_offnom_list = eg["complex_response_offnominal_list"]
    omega = eg["omega"]

    # Plot: Magnitude response of nominal and off-nominal systems
    fig, _ = dkpy.plot_magnitude_response_uncertain_model_set(
        response_nom,
        response_offnom_list,
        omega,
    )

    # Plot: Phase response of nominal and off-nominal systems
    fig, _ = dkpy.plot_phase_response_uncertain_model_set(
        response_nom,
        response_offnom_list,
        omega,
    )

    # Plot: Singular value response of nominal and off-nominal systems
    fig, ax = dkpy.plot_singular_value_response_uncertain_model_set(
        response_nom,
        response_offnom_list,
        omega,
    )

    # Uncertainty models
    uncertainty_models = {
        "additive",
        "multiplicative_input",
        "multiplicative_output",
        "inverse_additive",
        "inverse_multiplicative_input",
        "inverse_multiplicative_output",
    }
    response_residuals_dict = dkpy.compute_uncertainty_residual_response(
        response_nom,
        response_offnom_list,
        uncertainty_models,
    )

    # Plot: Singular value response of uncerainty residuals
    figure_dict = dkpy.plot_singular_value_response_residual(
        response_residuals_dict, omega
    )

    # Plot: Comparison of singular value response of uncerainty residuals for each
    # uncertainty model
    fig, _ = dkpy.plot_singular_value_response_residual_comparison(
        response_residuals_dict, omega
    )

    # Compute uncertainty weight frequency response
    response_weight_left, response_weight_right = (
        dkpy.compute_optimal_uncertainty_weight_response(
            response_residuals_dict["inverse_additive"], "diagonal", "diagonal"
        )
    )

    # Fit overbounding stable and minimum-phase uncertainty weight system
    weight_left = dkpy.fit_overbounding_uncertainty_weight(
        response_weight_left, omega, [4, 5]
    )
    weight_right = dkpy.fit_overbounding_uncertainty_weight(
        response_weight_right, omega, [3, 5]
    )

    # Plot: Magnitude response of uncertainty weight frequency response and overbounding
    # fit
    fig, _ = dkpy.plot_magnitude_response_uncertainty_weight(
        response_weight_left,
        response_weight_right,
        omega,
        weight_left,
        weight_right,
    )
    plt.show()


if __name__ == "__main__":
    example_uncertainty_characterization()
