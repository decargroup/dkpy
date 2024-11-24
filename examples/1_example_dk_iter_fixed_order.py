"""D-K iteration with fixed number of iterations and fit order."""

import control
import numpy as np
from matplotlib import pyplot as plt

import dkpy


def example_dk_iter_fixed_order():
    """D-K iteration with fixed number of iterations and fit order."""
    # Plant
    G0 = np.array(
        [
            [87.8, -86.4],
            [108.2, -109.6],
        ]
    )
    G = control.append(
        control.TransferFunction([1], [75, 1]),
        control.TransferFunction([1], [75, 1]),
    ) * control.TransferFunction(
        G0.reshape(2, 2, 1),
        np.ones((2, 2, 1)),
    )
    # Weights
    Wp = 0.5 * control.append(
        control.TransferFunction([10, 1], [10, 1e-5]),
        control.TransferFunction([10, 1], [10, 1e-5]),
    )
    Wi = control.append(
        control.TransferFunction([1, 0.2], [0.5, 1]),
        control.TransferFunction([1, 0.2], [0.5, 1]),
    )
    G.name = "G"
    Wp.name = "Wp"
    Wi.name = "Wi"
    sum_w = control.summing_junction(
        inputs=["u_w", "u_G"],
        dimension=2,
        name="sum_w",
    )
    sum_del = control.summing_junction(
        inputs=["u_del", "u_u"],
        dimension=2,
        name="sum_del",
    )
    split = control.summing_junction(
        inputs=["u"],
        dimension=2,
        name="split",
    )
    P = control.interconnect(
        syslist=[G, Wp, Wi, sum_w, sum_del, split],
        connections=[
            ["G.u", "sum_del.y"],
            ["sum_del.u_u", "split.y"],
            ["sum_w.u_G", "G.y"],
            ["Wp.u", "sum_w.y"],
            ["Wi.u", "split.y"],
        ],
        inplist=["sum_del.u_del", "sum_w.u_w", "split.u"],
        outlist=["Wi.y", "Wp.y", "-sum_w.y"],
    )
    # Dimensions
    n_y = 2
    n_u = 2

    dk_iter = dkpy.DkIterFixedOrder(
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
        n_iterations=3,
        fit_order=4,
    )

    omega = np.logspace(-3, 3, 61)
    block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    K, N, mu, d_scale_fit_info, info = dk_iter.synthesize(
        P,
        n_y,
        n_u,
        omega,
        block_structure,
    )

    print(mu)

    fig, ax = plt.subplots()
    for i, ds in enumerate(d_scale_fit_info):
        ds.plot_mu(ax=ax, plot_kw=dict(label=f"iter{i}"))

    ax = None
    for i, ds in enumerate(d_scale_fit_info):
        _, ax = ds.plot_D(ax=ax, plot_kw=dict(label=f"iter{i}"))

    plt.show()


if __name__ == "__main__":
    example_dk_iter_fixed_order()
