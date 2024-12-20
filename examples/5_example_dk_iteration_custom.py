"""D-K iteration with fixed number of iterations and fit order.

If you don't have access to MOSEK, see the ``DkIteration`` object settings from
the first two examples.
"""

import numpy as np
from matplotlib import pyplot as plt

import dkpy


class MyDkIter(dkpy.DkIteration):
    """Custom D-K iteration class with interactive order selection."""

    def __init__(
        self,
        controller_synthesis,
        structured_singular_value,
        d_scale_fit,
    ):
        super().__init__(
            controller_synthesis,
            structured_singular_value,
            d_scale_fit,
        )
        self.my_iter_count = 0

    def _get_fit_order(
        self,
        iteration,
        omega,
        mu_omega,
        D_omega,
        P,
        K,
        block_structure,
    ):
        print(f"Iteration {self.my_iter_count} with mu of {np.max(mu_omega)}")
        if self.my_iter_count < 3:
            self.my_iter_count += 1
            return 4
        else:
            return None


def example_dk_iter_custom():
    """D-K iteration with fixed number of iterations and fit order."""
    eg = dkpy.example_skogestad2006_p325()

    dk_iter = MyDkIter(
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
    example_dk_iter_custom()
