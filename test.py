import numpy as np
import control
from matplotlib import pyplot as plt

import dkpy


def main():
    # Frequency range
    omega_min = 0.1
    omega_max = 10
    num_omega = 100
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_omega)

    # Ground-truth system
    sys = control.TransferFunction([1, 2, 2], [1, 2.5, 1.5]) * control.TransferFunction(
        1, [1, 0.1]
    )
    sys = sys * control.TransferFunction([1, 3.75, 3.5], [1, 2.5, 13])

    # System frequency response
    frd_sys = control.frequency_response(sys, omega, squeeze=False)
    fresp_sys = frd_sys.complex
    mag_sys = np.array(frd_sys.magnitude[0, 0, :])
    # mag_sys = mag_sys * (1 + np.random.normal(0.0, 0.05, mag_sys.size))

    order = 4
    mosek_params = {
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_INFEAS": 1e-12,
    }
    linear_solver_param = {
        "solver": "MOSEK",
        "verbose": False,
        "mosek_params": mosek_params,
        "warm_start": True,
    }

    weight_log_cheby = dkpy._fit_magnitude_log_chebyshev_siso(
        omega,
        mag_sys,
        order,
    )

    # Estimated Frequency response
    omega_fit = np.logspace(np.log10(omega_min), np.log10(omega_max), 100)
    frd_fit = weight_log_cheby.frequency_response(omega_fit)
    mag_fit = frd_fit.magnitude

    # Plot: Preview log-magnitude response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, control.mag2db(mag_sys), "--*", label="Magnitude Data")
    ax.semilogx(
        omega_fit, control.mag2db(mag_fit), label=f"Fit Magnitude (Order {order})"
    )
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    # Plot: Preview magnitude response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, mag_sys, "--*", label="Magnitude Data")
    ax.semilogx(omega_fit, mag_fit, label=f"Fit Magnitude (Order {order})")
    ax.set_ylabel("Magnitude (-)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    # Plot: Preview residual response
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogx(omega, mag_sys**2 / mag_fit**2 - 1, "--*", label="Magnitude Data")
    ax.set_ylabel("Residual (-)")
    ax.set_xlabel("$f$ (Hz)")
    ax.legend()
    # fig.savefig(f"figs/log_cheby_fit_order_{order}.pdf")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
