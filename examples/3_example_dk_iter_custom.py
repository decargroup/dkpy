"""D-K iteration with fixed number of iterations and fit order."""

import control
import numpy as np
from matplotlib import pyplot as plt

import dkpy


class MyDkIter(dkpy.DkIteration):
    """Custom D-K iteration class with interactive order selection."""

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
        d_info = []
        for fit_order in range(5):
            D_fit, D_fit_inv = self.transfer_function_fit.fit(
                omega,
                D_omega,
                order=fit_order,
                block_structure=block_structure,
            )
            d_info.append(
                dkpy.DScaleFitInfo.create_from_fit(
                    omega,
                    mu_omega,
                    D_omega,
                    P,
                    K,
                    D_fit,
                    D_fit_inv,
                    block_structure,
                )
            )
        fig, ax = plt.subplots()
        dkpy.plot_mu(d_info[0], ax=ax, plot_kw=dict(label="true"), hide="mu_fit_omega")
        for i, ds in enumerate(d_info):
            dkpy.plot_mu(ds, ax=ax, plot_kw=dict(label=f"order={i}"), hide="mu_omega")
        print("Close plot to continue...")
        plt.show()
        selected_order_str = input("Select order (<Enter> to end iteration): ")
        if selected_order_str == "":
            return None
        else:
            return int(selected_order_str)


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
        transfer_function_fit=dkpy.TfFitSlicot(),
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

    print(mu)

    fig, ax = plt.subplots()
    for i, ds in enumerate(d_scale_fit_info):
        dkpy.plot_mu(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    ax = None
    for i, ds in enumerate(d_scale_fit_info):
        _, ax = dkpy.plot_D(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    plt.show()


if __name__ == "__main__":
    example_dk_iter_custom()
