"""D-K iteration with nonsquare perturbations

D-K iteration with a fixed number of iterations and fit order and closed-loop
simluation of perturbed models for the linearized lateral dynamics of an
aircraft.

The quantities of interest in the aircraft dynamics are shown below.

Airframe States
---------------
phi: Roll angle
beta: Sideslip angle
p: Roll rate
r: Yaw rate

Control Inputs
--------------
zeta: Rudder angle input
xi: Aileron angle input

Disturbances
------------
beta_w: Sideslip angle wind disturbance
p_w: Roll rate wind disturbance

Tracking Reference
------------------
phi_ref: Roll angle reference
"""

import numpy as np
import control
from matplotlib import pyplot as plt

import dkpy


def example_dk_iter_list_order_aircraft():
    # Example parameters
    eg = dkpy.utilities.example_mackenroth2004_p430()
    plant_gen = eg["P"]

    n_u = eg["n_u"]
    n_y = eg["n_y"]
    n_u_delta = eg["n_u_delta"]
    n_y_delta = eg["n_y_delta"]
    n_w = eg["n_w"]
    n_z = eg["n_z"]

    # Frequency
    freq_min = 0.01
    freq_max = 100
    num_freq = 100
    freq = np.logspace(np.log10(freq_min), np.log10(freq_max), num_freq)
    omega = 2 * np.pi * freq

    # DK-iteration controller synthesis
    dk_iter = dkpy.DkIterListOrder(
        controller_synthesis=dkpy.HinfSynLmi(
            lmi_strictness=5e-7,  # Have to play with tolerances.
            solver_params=dict(
                solver="MOSEK",
                eps=5e-8,
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
        fit_orders=[4, 4],
    )
    # Alternative MATLAB block structure description
    # block_structure = np.array([[n_u_delta, n_y_delta], [n_w, n_z]])
    block_structure = [
        dkpy.ComplexFullBlock(n_y_delta, n_u_delta),
        dkpy.ComplexFullBlock(n_w, n_z),
    ]
    controller, N, mu, iter_results, info = dk_iter.synthesize(
        plant_gen,
        n_y,
        n_u,
        omega,
        block_structure,
    )
    controller.set_inputs(["phi_ref", "phi_meas", "beta_meas", "p_meas", "r_meas"])
    controller.set_outputs(["zeta_c", "xi_c"])
    print("mu:", mu)

    # Plot - DK-iteration results
    fig, ax = plt.subplots()
    for i, ds in enumerate(iter_results):
        dkpy.plot_mu(ds, ax=ax, plot_kw=dict(label=f"iter{i}"))

    # Generate off-nominal models with different perturbations
    delta_block_list = []
    # Nominal system uncertainty (no perturbation)
    delta_block = control.StateSpace([], [], [], [[0, 0], [0, 0]])
    delta_block.set_inputs(delta_block.ninputs, "y_del")
    delta_block.set_outputs(delta_block.noutputs, "u_del")
    delta_block_list.append(delta_block)
    # Constant gain perturbation
    num_offnom_gain = 20
    for gain_unc in np.linspace(-1, 1, num_offnom_gain):
        delta = control.StateSpace([], [], [], [gain_unc])
        delta_block = control.append(delta, delta)
        delta_block.set_inputs(delta_block.ninputs, "y_del")
        delta_block.set_outputs(delta_block.noutputs, "u_del")
        delta_block_list.append(delta_block)
    # Phase perturbation
    a_min = 0.01
    a_max = 10
    num_offnom_phase = 20
    for a in np.linspace(a_min, a_max, num_offnom_phase):
        delta_block = control.append(
            control.tf([1 / a, -1], [1 / a, 1]),
            control.tf([a, -1], [a, 1]),
            name="delta_block",
        )
        delta_block.set_inputs(delta_block.ninputs, "y_del")
        delta_block.set_outputs(delta_block.noutputs, "u_del")
        delta_block_list.append(delta_block)

    # Closed-loop simulation system
    input_id_sim_list = [
        "phi_ref",
        "beta_w",
        "p_w",
        "n[0]",
        "n[1]",
        "n[2]",
        "n[3]",
    ]
    output_id_sim_list = [
        "phi",
        "beta",
        "p",
        "r",
        "zeta_c",
        "xi_c",
        "zeta",
        "xi",
        "rate_xi",
        "rate_zeta",
    ]

    # Uncertain closed-loop system
    airframe = eg["airframe"]
    actuator = eg["actuator"]
    weight_unc = eg["weight_unc"]
    sum_noise = eg["sum_noise"]
    sum_uncertainty = eg["sum_uncertainty"]
    closed_loop_sim_list = []
    for delta_block in delta_block_list:
        closed_loop_sim = control.interconnect(
            syslist=[
                airframe,
                actuator,
                controller,
                weight_unc,
                delta_block,
                sum_noise,
                sum_uncertainty,
            ],
            inplist=input_id_sim_list,
            outlist=output_id_sim_list,
            name="closed_loop_sim",
        )
        closed_loop_sim.set_inputs(input_id_sim_list)
        closed_loop_sim.set_outputs(output_id_sim_list)
        closed_loop_sim_list.append(closed_loop_sim)

    # Time
    time_min = 0
    time_max = 7.5
    dt = 0.01
    num_time = round((time_max - time_min) / dt)
    time = np.arange(time_min, time_max, dt)
    # Reference signal
    phi_ref = 6 * np.ones_like(time)
    # Disturbance signals
    beta_w = np.zeros_like(time)
    p_w = np.zeros_like(time)
    # Sensor noise signals (assume sensors have identical noise characteristics)
    mean_noise = 0
    std_noise = 0.005
    noise_phi = np.random.normal(mean_noise, std_noise, num_time)
    noise_beta = np.random.normal(mean_noise, std_noise, num_time)
    noise_p = np.random.normal(mean_noise, std_noise, num_time)
    noise_r = np.random.normal(mean_noise, std_noise, num_time)
    # Exogenous input signal
    inputs_sim = np.array(
        [phi_ref, beta_w, p_w, noise_phi, noise_beta, noise_p, noise_r]
    )

    # Simulation response
    trd_sim_list = control.forced_response(
        sysdata=closed_loop_sim_list, T=time, U=inputs_sim
    )

    # Plot - Reference signal
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, phi_ref)
    ax.set_ylabel("$\\phi_r$ (deg)")
    ax.set_xlabel("$t$ (s)")
    ax.grid(linestyle="--")

    # Plot - Disturbance signals
    fig, ax = plt.subplots(2, layout="constrained", sharex=True)
    ax[0].plot(time, beta_w)
    ax[1].plot(time, p_w)
    ax[0].set_ylabel("$\\beta_w$ (deg)")
    ax[0].grid(linestyle="--")
    ax[1].set_ylabel("$p_w$ (deg)")
    ax[1].grid(linestyle="--")
    ax[-1].set_xlabel("$t$ (s)")
    fig.align_ylabels()

    # Plot - Noise signals
    fig, ax = plt.subplots(4, layout="constrained", sharex=True)
    ax[0].plot(time, noise_phi)
    ax[1].plot(time, noise_beta)
    ax[2].plot(time, noise_p)
    ax[3].plot(time, noise_r)
    ax[0].set_ylabel("$n_{\\phi}$ (deg)")
    ax[1].set_ylabel("$n_{\\beta}$ (deg)")
    ax[2].set_ylabel("$n_{p}$ (deg/s)")
    ax[3].set_ylabel("$n_{r}$ (deg/s)")
    ax[-1].set_xlabel("$t$ (s)")
    for a in ax:
        a.grid(linestyle="--")
    fig.align_ylabels()

    # Plot - Output signals
    fig, ax = plt.subplots(4, layout="constrained", sharex=True)
    # Reference signal
    ax[0].plot(time, phi_ref, color="black", linestyle="--", label="Reference")
    for idx_sim, trd_sim in enumerate(reversed(trd_sim_list)):
        # Output signals
        phi = trd_sim.y[0, :]
        beta = trd_sim.y[1, :]
        p = trd_sim.y[2, :]
        r = trd_sim.y[3, :]
        # Plot signals (nominal)
        if idx_sim == len(trd_sim_list) - 1:
            ax[0].plot(time, phi, color="tab:blue", label="Nominal")
            ax[1].plot(time, beta, color="tab:blue", label="Nominal")
            ax[2].plot(time, p, color="tab:blue", label="Nominal")
            ax[3].plot(time, r, color="tab:blue", label="Nominal")
        # Plot signals (off-nominal)
        else:
            ax[0].plot(time, phi, color="tab:orange", alpha=0.25, label="Off-Nominal")
            ax[1].plot(time, beta, color="tab:orange", alpha=0.25, label="Off-Nominal")
            ax[2].plot(time, p, color="tab:orange", alpha=0.25, label="Off-Nominal")
            ax[3].plot(time, r, color="tab:orange", alpha=0.25, label="Off-Nominal")
    ax[0].set_ylabel("$\\phi$ (deg)")
    ax[1].set_ylabel("$\\beta$ (deg)")
    ax[2].set_ylabel("$p$ (deg/s)")
    ax[3].set_ylabel("$r$ (deg/s)")
    ax[-1].set_xlabel("$t$ (s)")
    for a in ax:
        a.grid(linestyle="--")
    handles, labels = ax[0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.align_ylabels()
    fig.legend(
        handles=legend_dict.values(),
        labels=legend_dict.keys(),
        loc="outside lower center",
        ncol=3,
    )

    # Plot - Input signals
    fig, ax = plt.subplots(4, layout="constrained", sharex=True)
    for idx_sim, trd_sim in enumerate(reversed(trd_sim_list)):
        # Control signals
        zeta = trd_sim.y[6, :]
        xi = trd_sim.y[7, :]
        rate_zeta = trd_sim.y[8, :]
        rate_xi = trd_sim.y[9, :]
        if idx_sim == len(trd_sim_list) - 1:
            ax[0].plot(
                time, zeta, color="tab:blue", linestyle="-", label="Actuator (Nominal)"
            )
            ax[1].plot(
                time,
                rate_zeta,
                color="tab:blue",
                linestyle="-",
                label="Actuator (Nominal)",
            )
            ax[2].plot(
                time, xi, color="tab:blue", linestyle="-", label="Actuator (Nominal)"
            )
            ax[3].plot(
                time,
                rate_xi,
                color="tab:blue",
                linestyle="-",
                label="Actuator (Nominal)",
            )
        else:
            ax[0].plot(
                time,
                zeta,
                color="tab:orange",
                alpha=0.25,
                linestyle="-",
                label="Actuator (Off-Nominal)",
            )
            ax[1].plot(
                time,
                rate_zeta,
                color="tab:orange",
                alpha=0.25,
                linestyle="-",
                label="Actuator (Off-Nominal)",
            )
            ax[2].plot(
                time,
                xi,
                color="tab:orange",
                alpha=0.25,
                linestyle="-",
                label="Actuator (Off-Nominal)",
            )
            ax[3].plot(
                time,
                rate_xi,
                color="tab:orange",
                alpha=0.25,
                linestyle="-",
                label="Actuator (Off-Nominal)",
            )
    ax[0].set_ylabel("$\\zeta$ (deg)")
    ax[1].set_ylabel("$\\xi$ (deg)")
    ax[2].set_ylabel("$\\dot{\\zeta}$ (deg/s)")
    ax[3].set_ylabel("$\\dot{\\xi}$ (deg/s)")
    ax[-1].set_xlabel("$t$ (s)")
    for a in ax:
        a.grid(linestyle="--")
    handles, labels = ax[0].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    fig.align_ylabels()
    fig.legend(
        handles=legend_dict.values(),
        labels=legend_dict.keys(),
        loc="outside lower center",
        ncol=2,
    )
    plt.show()


if __name__ == "__main__":
    example_dk_iter_list_order_aircraft()
