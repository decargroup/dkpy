"""D-K iteration with fixed number of iterations and fit order."""

import numpy as np
from matplotlib import pyplot as plt

import dkpy


def example_dk_iter_list_order():
    """D-K iteration with a list of fit orders."""
    eg = dkpy.example_skogestad2006_p325()

    dk_iter = dkpy.DkIterListOrder(
        controller_synthesis=dkpy.HinfSynSlicot(),
        structured_singular_value=dkpy.SsvLmiBisection(),
        d_scale_fit=dkpy.DScaleFitSlicot(),
        # fit_orders=[4, 4, 4],  # <- an alternative
        fit_orders=[
            np.diag([4, 4, 0, 0]),
            np.diag([4, 4, 0, 0]),
            np.diag([4, 4, 0, 0]),
        ],
    )

    omega = np.logspace(-3, 3, 61)
    block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    K, N, mu, iter_results, info = dk_iter.synthesize(
        eg["P"],
        eg["n_y"],
        eg["n_u"],
        omega,
        block_structure,
    )

    print(f"mu={mu}")

    fig, ax = plt.subplots()
    for i, ds in enumerate(iter_results):
        dkpy.plot_mu(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    ax = None
    for i, ds in enumerate(iter_results):
        _, ax = dkpy.plot_D(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    plt.show()


if __name__ == "__main__":
    example_dk_iter_list_order()
