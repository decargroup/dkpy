"""D-K iteration with fixed number of iterations and fit order."""

import logging

import numpy as np
from matplotlib import pyplot as plt

import dkpy

logging.basicConfig(level=logging.INFO)


def example_dk_iter_auto_order():
    """D-K iteration automatically selected fit orders."""
    eg = dkpy.example_skogestad2006_p325()

    dk_iter = dkpy.DkIterAutoOrder(
        controller_synthesis=dkpy.HinfSynLmi(
            lmi_strictness=1e-7,
            solver_params=dict(
                solver="MOSEK",
                eps=1e-8,
            ),
        ),
        structured_singular_value=dkpy.SsvLmiBisection(
            bisection_atol=1e-5,
            bisection_rtol=1e-5,
            max_iterations=1000,
            lmi_strictness=1e-7,
            solver_params=dict(
                solver="MOSEK",
                eps=1e-9,
            ),
        ),
        d_scale_fit=dkpy.DScaleFitSlicot(),
        max_mu=1,
        max_mu_fit_error=1e-2,
        max_iterations=3,
        max_fit_order=4,
    )

    omega = np.logspace(-3, 3, 61)
    block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    K, N, mu, d_scale_fit_info, info = dk_iter.synthesize(
        eg["P"],
        eg["n_y"],
        eg["n_u"],
        omega,
        block_structure,
    )

    print(f"mu={mu}")

    fig, ax = plt.subplots()
    for i, ds in enumerate(d_scale_fit_info):
        dkpy.plot_mu(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    ax = None
    for i, ds in enumerate(d_scale_fit_info):
        _, ax = dkpy.plot_D(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    plt.show()


if __name__ == "__main__":
    example_dk_iter_auto_order()