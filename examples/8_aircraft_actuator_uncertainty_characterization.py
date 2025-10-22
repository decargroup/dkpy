"""Multi-model aircraft actuator uncertainty characterization from frequency response
data.
"""

from matplotlib import pyplot as plt

import dkpy


def example_aircraft_uncertainty_characterization():
    """Multi-model aircraft actuator uncertainty characterization from frequency
    response data.
    """
    # Load example data
    eg = dkpy.example_aircraft_actuator_uncertainty()
    response_actuator = eg["response_actuator_nominal"]
    response_actuator_offnom_list = eg["response_actuator_offnominal"]
    omega = eg["omega"]

    # Compute the residual response for different uncertainty models
    complex_response_residual_dict = dkpy.compute_uncertainty_residual_response(
        response_actuator,
        response_actuator_offnom_list,
        {"additive", "multiplicative_input", "inverse_multiplicative_input"},
    )

    # Compute the optimal uncertainty weights with a given structure
    response_weight_left, response_weight_right = (
        dkpy.compute_uncertainty_weight_response(
            complex_response_residual_dict["multiplicative_input"],
            "scalar",
            "identity",
        )
    )

    # Fit an overbounding LTI system to the optimal uncertainty weight response
    weight_left = dkpy.fit_uncertainty_weight(response_weight_left, omega, 1)

    # Plot: Singular value response of nominal and off-nominal systems
    fig, _, _ = dkpy.plot_singular_value_response_uncertain_model_set(
        response_actuator, response_actuator_offnom_list, omega, hz=True
    )
    fig.savefig("singular_value_plot.png")

    # Plot: Comparison of singular value response of uncerainty residuals for each
    # uncertainty model
    fig, _, _ = dkpy.plot_singular_value_response_residual_comparison(
        complex_response_residual_dict, omega, hz=True
    )
    fig.savefig("residual_comparison.png")

    # Plot: Magnitude response of uncertainty weight frequency response and overbounding
    # fit
    fig, _, _ = dkpy.plot_magnitude_response_uncertainty_weight(
        response_weight_left,
        response_weight_right,
        omega,
        weight_left=weight_left,
        hz=True,
    )
    fig.savefig("weight.png")
    plt.show()


if __name__ == "__main__":
    example_aircraft_uncertainty_characterization()
