"""Visualization helpers for FLIM analysis results."""

import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes


def create_FLIM_image(
    mean_photon_arrival_time,
    intensity,
    colormap=cm.rainbow,
    lt_min=None,
    lt_max=None,
    int_min=None,
    int_max=None,
):
    """Create an RGB FLIM image from lifetime and intensity arrays."""
    if mean_photon_arrival_time.shape != intensity.shape:
        raise ValueError("Lifetime and intensity arrays must have the same shape")
    if lt_min is None or lt_max is None:
        lt_min = np.nanmin(mean_photon_arrival_time)
        lt_max = np.nanmax(mean_photon_arrival_time)
    if lt_max == lt_min:
        raise ValueError(f"lt_max and lt_min must differ - got {lt_min}")
    if int_min is None or int_max is None:
        int_min = np.nanmin(intensity)
        int_max = np.nanmax(intensity)
    if int_max == int_min:
        raise ValueError("int_max and int_min must differ")
    lifetime_normalized = np.clip(
        (mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1
    )
    lifetime_rgb = colormap(lifetime_normalized)[..., :3]
    intensity_normalized = np.clip(
        (intensity - int_min) / (int_max - int_min), 0, 1
    )
    return lifetime_rgb * intensity_normalized[..., np.newaxis]


def draw_unitary_circle(
    ax: Axes,
    sync_rate,
    tau_max: int = None,
    tick_length=0.02,
    color='white',
    label_color='white',
    xlim=None,
    ylim=None,
):
    if not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes object, got {type(ax).__name__}")
    
    def inside_limits(point, xlim, ylim, margin=0):
        x, y = point
        return (
            xlim[0] - margin <= x <= xlim[1] + margin and
            ylim[0] - margin <= y <= ylim[1] + margin
        )
    
    # determine limits first
    if xlim is None:
        xlim = (-0.1, 1.1)

    if ylim is None:
        ylim = (0, 0.8)


    omega = 2 * np.pi * sync_rate

    if tau_max is None:
        period_ns = 1 / sync_rate
        tau_max = int(np.ceil(period_ns / 2))
        
    # taus_ns = np.arange(1, tau_max + 1)
    ticks = np.arange(1, tau_max + 1)
    taus_ns = ticks[(ticks <= 8) | ((ticks > 8) & (ticks % 2 == 0))]

    center = np.array([0.5, 0])
    radius = 0.5

    # Generate circle points
    theta = np.linspace(0, np.pi, 300)
    g_circle = center[0] + radius * np.cos(theta)
    s_circle = center[1] + radius * np.sin(theta)
    ax.plot(g_circle, s_circle, '-', color=color, label='Universal Circle', lw=1)

    # Phasor function
    def phasor(tau):
        g = 1 / (1 + (omega * tau)**2)
        s = (omega * tau) / (1 + (omega * tau)**2)
        return np.array([g, s])

    for tau_ns in taus_ns:
        tau_s = tau_ns * 1e-9
        p = phasor(tau_s)

        v = p - center
        v_unit = v / np.linalg.norm(v)

        p1 = p - (tick_length / 2) * v_unit
        p2 = p + (tick_length / 2) * v_unit

        label_pos = p + (tick_length * 1.2) * v_unit

        # draw tick only if visible
        if inside_limits(p1, xlim, ylim) or inside_limits(p2, xlim, ylim):
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                '-',
                color=color,
                lw=1,
            )

        # draw label only if visible
        if inside_limits(label_pos, xlim, ylim):
            ax.text(
                label_pos[0],
                label_pos[1],
                f'{tau_ns}',
                fontsize=8,
                ha='center',
                va='center',
                color=label_color,
            )
            
    ax.set_xlabel('g')
    ax.set_ylabel('s')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    # Clean formatting
    ax.set_aspect('equal', adjustable = 'datalim')

