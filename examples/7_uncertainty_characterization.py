"""Multi-model uncertainty characterization."""

import control
import numpy as np
from matplotlib import pyplot as plt

import dkpy
from dkpy.uncertainty_characterization import plot_magnitude_response_nom_offnom


def example_uncertainty_characterization():
    """Multi-model uncertainty characterization."""
    # Nominal model paramters
    omega_n_11_nom = 5
    omega_n_22_nom = 1
    gamma_11_nom = 0.4
    gamma_22_nom = 0.6
    gain_12_nom = 3
    gain_21_nom = 10
    tau_12_nom = 0.1
    tau_21_nom = 0.5
    # Generate nominal model
    sys_nom = generate_sys(
        omega_n_11_nom,
        omega_n_22_nom,
        gamma_11_nom,
        gamma_22_nom,
        gain_12_nom,
        gain_21_nom,
        tau_12_nom,
        tau_21_nom,
    )

    # Off-nominal system relative parameter variation
    rel_variation_omega_n = 0.20
    rel_variation_gamma = 0.20
    rel_variation_gain = 0.05
    rel_variation_tau = 0.25
    num_offnom = 50

    # Generate off-nominal models
    np.random.seed(0)
    sys_offnom_list = []
    for _ in range(num_offnom):
        # Off-nominal system paramters
        omega_n_11_offnom = omega_n_11_nom * (
            1 + rel_variation_omega_n * (2 * np.random.rand() - 1)
        )
        omega_n_22_offnom = omega_n_22_nom * (
            1 + rel_variation_omega_n * (2 * np.random.rand() - 1)
        )
        gamma_11_offnom = gamma_11_nom * (
            1 + rel_variation_gamma * (2 * np.random.rand() - 1)
        )
        gamma_22_offnom = gamma_22_nom * (
            1 + rel_variation_gamma * (2 * np.random.rand() - 1)
        )
        gain_12_offnom = gain_12_nom * (
            1 + rel_variation_gain * (2 * np.random.rand() - 1)
        )
        gain_21_offnom = gain_21_nom * (
            1 + rel_variation_gain * (2 * np.random.rand() - 1)
        )
        tau_12_offnom = tau_12_nom * (
            1 + rel_variation_tau * (2 * np.random.rand() - 1)
        )
        tau_21_offnom = tau_21_nom * (
            1 + rel_variation_tau * (2 * np.random.rand() - 1)
        )
        # Generate off-nominal system
        sys_offnom = generate_sys(
            omega_n_11_offnom,
            omega_n_22_offnom,
            gamma_11_offnom,
            gamma_22_offnom,
            gain_12_offnom,
            gain_21_offnom,
            tau_12_offnom,
            tau_21_offnom,
        )
        sys_offnom_list.append(sys_offnom)

    # Frequency grid
    omega_min = 0.05
    omega_max = 50
    num_omega = 100
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_omega)

    # Nominal and off-nominal frequency response
    frequency_response_nom = control.frequency_response(sys_nom, omega)
    frequency_response_offnom_list = control.frequency_response(sys_offnom_list, omega)

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
            uncertainty_residuals_dict["I"], "diagonal", "diagonal"
        )
    )

    # Plot:
    dkpy.plot_magnitude_response_uncertainty_weight(
        response_weight_left, response_weight_right
    )
    plt.show()


def generate_sys(
    omega_n_11, omega_n_22, gamma_11, gamma_22, gain_12, gain_21, tau_12, tau_21
):
    """Generate example transfer function system from parameters."""
    tf_11 = control.TransferFunction(
        [omega_n_11**2],
        [1, 2 * gamma_11 * omega_n_11, omega_n_11**2],
    )
    tf_12 = control.TransferFunction([gain_12], [tau_12, 1])
    tf_21 = control.TransferFunction([gain_21], [tau_21, 1])
    tf_22 = control.TransferFunction(
        [omega_n_22**2],
        [1, 2 * gamma_22 * omega_n_22, omega_n_22**2],
    )
    tf = control.combine_tf([[tf_11, tf_12], [tf_21, tf_22]])

    return tf


if __name__ == "__main__":
    example_uncertainty_characterization()
