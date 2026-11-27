"""Visualization helpers for FLIM analysis results.

Requires matplotlib, an optional dependency: ``pip install tttrkit[plot]``.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.axes import Axes
except ImportError as e:
    raise ImportError(
        "tttrkit.analysis.visualization requires matplotlib; install it "
        "with `pip install tttrkit[plot]`."
    ) from e



def create_flim_image(
    lifetime,
    intensity,
    colormap=cm.rainbow,
    lifetime_min=None,
    lifetime_max=None,
    intensity_min=None,
    intensity_max=None,
):
    """RGB FLIM image: hue from ``lifetime``, brightness from ``intensity``.

    Args:
        lifetime: Per-pixel lifetime (or mean arrival time), any unit.
        intensity: Per-pixel intensity, same shape.
        colormap: Matplotlib colormap for the lifetime.
        lifetime_min, lifetime_max: Lifetime mapped to the ends of the
            colormap (default: the data range).
        intensity_min, intensity_max: Intensity mapped to black and full
            brightness (default: the data range).
    """
    if lifetime.shape != intensity.shape:
        raise ValueError("Lifetime and intensity arrays must have the same shape")
    if lifetime_min is None or lifetime_max is None:
        lifetime_min = np.nanmin(lifetime)
        lifetime_max = np.nanmax(lifetime)
    if lifetime_max == lifetime_min:
        raise ValueError(
            f"lifetime_max and lifetime_min must differ - got {lifetime_min}"
        )
    if intensity_min is None or intensity_max is None:
        intensity_min = np.nanmin(intensity)
        intensity_max = np.nanmax(intensity)
    if intensity_max == intensity_min:
        raise ValueError("intensity_max and intensity_min must differ")
    lifetime_normalized = np.clip(
        (lifetime - lifetime_min) / (lifetime_max - lifetime_min), 0, 1
    )
    lifetime_rgb = colormap(lifetime_normalized)[..., :3]
    intensity_normalized = np.clip(
        (intensity - intensity_min) / (intensity_max - intensity_min), 0, 1
    )
    return lifetime_rgb * intensity_normalized[..., np.newaxis]


def draw_universal_circle(
    ax: Axes,
    sync_rate: float,
    tau_max_ns: int | None = None,
    tick_length=0.02,
    color='white',
    label_color='white',
    xlim=None,
    ylim=None,
):
    """Draw the universal circle of the phasor plot with lifetime ticks.

    Args:
        ax: Axes to draw in.
        sync_rate: Hz; sets the phasor frequency.
        tau_max_ns: Largest lifetime tick, in ns (default: half the sync
            period). Ticks every ns up to 8 ns, every 2 ns above.
    """
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

    if tau_max_ns is None:
        period_ns = 1e9 / sync_rate
        tau_max_ns = int(np.ceil(period_ns / 2))

    ticks = np.arange(1, tau_max_ns + 1)
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


def plot_decay(decays, labels, time_axis=None, title=None):
    """Plot decays on a log scale.

    Args:
        decays: Sequence of 1-D decays (counts per TCSPC channel).
        labels: One legend label per decay.
        time_axis: Seconds; one array for all decays, a sequence of one
            array per decay, or None to plot against the channel index.
            Plotted in ns.
        title: Optional axes title.
    """
    if len(decays) != len(labels):
        raise ValueError(
            f"Mismatch between number of decays ({len(decays)}) and labels "
            f"({len(labels)})"
        )

    fig, ax = plt.subplots(figsize=(6, 4))

    # One time axis per decay, or one shared by all
    per_decay = (
        isinstance(time_axis, (list, tuple))
        and len(time_axis) > 0
        and hasattr(time_axis[0], "__len__")
    )
    if per_decay and len(time_axis) != len(decays):
        raise ValueError(
            f"Mismatch between number of time axes ({len(time_axis)}) and "
            f"decays ({len(decays)})"
        )

    for i, (decay, label) in enumerate(zip(decays, labels)):
        if time_axis is None:
            x = np.arange(len(decay))
        else:
            x = np.asarray(time_axis[i] if per_decay else time_axis) * 1e9
        ax.semilogy(x, decay, linestyle="", marker=".", markersize=2, label=label)

    ax.set_xlabel("time (ns)" if time_axis is not None else "channel")
    ax.set_ylabel("counts")
    if title:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_decay_fit(time_axis, decay, fit, residuals, fit_limits_channels, title=None):
    """Plot a decay with its fit, and the residuals below.

    Args:
        time_axis: Seconds, one value per channel of ``decay``; plotted in ns.
        decay: Measured counts per channel.
        fit: Fitted counts over the fit range.
        residuals: Residuals over the fit range.
        fit_limits_channels: (first, last) channel of the fit range, as
            returned by the decay fitters under "limits".
        title: Optional title.
    """
    time_ns = np.asarray(time_axis) * 1e9
    fit_time_ns = time_ns[fit_limits_channels[0]:fit_limits_channels[-1]]
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), height_ratios=[4, 1])
    ax[0].semilogy(time_ns, decay, color="blue", linestyle="", marker=".", markersize=2)
    ax[0].semilogy(fit_time_ns, fit, color="red")
    ax[0].set_ylabel("counts")
    ax[0].set_xticklabels("")
    if title:
        ax[0].set_title(title)

    ax[1].plot(fit_time_ns, residuals, color="blue", linestyle="-", linewidth=0.75, marker="")
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_xlabel("time (ns)")

    plt.show()
