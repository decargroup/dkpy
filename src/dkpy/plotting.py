"""Plotting utilities."""

__all__ = [
    "plot_D",
    "plot_mu",
]

from typing import Any, Dict, Tuple, Optional
import numpy as np
from matplotlib import pyplot as plt

from . import dk_iteration, fit_transfer_functions


def plot_mu(
    d_scale_info: dk_iteration.DScaleFitInfo,
    ax: Optional[plt.Axes] = None,
    plot_kw: Optional[Dict[str, Any]] = None,
    hide: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot mu.

    Parameters
    ----------
    d_scale_fit_info : dkpy.DScaleFitInfo
        Object containing information about the D-scale fit.
    ax : Optional[plt.Axes]
        Matplotlib axes to use.
    plot_kw : Optional[Dict[str, Any]]
        Keyword arguments for :func:`plt.Axes.semilogx`.
    hide : Optional[str]
        Set to ``'mu_omega'`` or ``'mu_fit_omega'`` to hide either one of
        those lines.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib :class:`plt.Figure` and :class:`plt.Axes` objects.
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Set label
    label = plot_kw.pop("label", "mu")
    label_mu_omega = label + ""
    label_mu_fit_omega = label + "_fit"
    # Clear line styles
    _ = plot_kw.pop("ls", None)
    _ = plot_kw.pop("linestyle", None)
    # Plot mu
    if hide != "mu_omega":
        ax.semilogx(
            d_scale_info.omega,
            d_scale_info.mu_omega,
            label=label_mu_omega,
            ls="--",
            **plot_kw,
        )
    if hide != "mu_fit_omega":
        ax.semilogx(
            d_scale_info.omega,
            d_scale_info.mu_fit_omega,
            label=label_mu_fit_omega,
            **plot_kw,
        )
    # Set axis labels
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel(r"$\mu(\omega)$")
    ax.grid(linestyle="--")
    ax.legend(loc="lower left")
    # Return figure and axes
    return fig, ax


def plot_D(
    d_scale_info: dk_iteration.DScaleFitInfo,
    ax: Optional[np.ndarray] = None,
    plot_kw: Optional[Dict[str, Any]] = None,
    hide: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot D.

    Parameters
    ----------
    d_scale_fit_info : dkpy.DScaleFitInfo
        Object containing information about the D-scale fit.
    ax : Optional[np.ndarray]
        Array of Matplotlib axes to use.
    plot_kw : Optional[Dict[str, Any]]
        Keyword arguments for :func:`plt.Axes.semilogx`.
    hide : Optional[str]
        Set to ``'D_omega'`` or ``'D_fit_omega'`` to hide either one of
        those lines.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Matplotlib :class:`plt.Figure` object and two-dimensional array of
        :class:`plt.Axes` objects.
    """
    mask = fit_transfer_functions._mask_from_block_structure(
        d_scale_info.block_structure
    )
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(
            mask.shape[0],
            mask.shape[1],
            constrained_layout=True,
        )
    else:
        fig = ax[0, 0].get_figure()
    # Set label
    label = plot_kw.pop("label", "D")
    label_D_omega = label + ""
    label_D_fit_omega = label + "_fit"
    # Clear line styles
    _ = plot_kw.pop("ls", None)
    _ = plot_kw.pop("linestyle", None)
    # Plot D
    mag_D_omega = np.abs(d_scale_info.D_omega)
    mag_D_fit_omega = np.abs(d_scale_info.D_fit_omega)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if mask[i, j] != 0:
                dB_omega = 20 * np.log10(mag_D_omega[i, j, :])
                dB_fit_omega = 20 * np.log10(mag_D_fit_omega[i, j, :])
                if hide != "D_omega":
                    ax[i, j].semilogx(
                        d_scale_info.omega,
                        dB_omega,
                        label=label_D_omega,
                        ls="--",
                        **plot_kw,
                    )
                if hide != "D_fit_omega":
                    ax[i, j].semilogx(
                        d_scale_info.omega,
                        dB_fit_omega,
                        label=label_D_fit_omega,
                        **plot_kw,
                    )
                # Set axis labels
                ax[i, j].set_xlabel(r"$\omega$ (rad/s)")
                ax[i, j].set_ylabel(rf"$D_{{{i}{j}}}(\omega) (dB)$")
                ax[i, j].grid(linestyle="--")
            else:
                ax[i, j].axis("off")
    fig.legend(handles=ax[0, 0].get_lines(), loc="lower left")
    # Return figure and axes
    return fig, ax
