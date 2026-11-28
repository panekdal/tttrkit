"""Plotting helpers (headless)."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tttrkit.analysis.visualization import draw_universal_circle  # noqa: E402


def test_universal_circle_ticks_default_to_half_the_sync_period():
    # 80 MHz: 12.5 ns period, ticks up to 7 ns (each labelled)
    fig, ax = plt.subplots()
    draw_universal_circle(ax, sync_rate=80e6, xlim=(-1, 2), ylim=(-1, 2))
    labels = sorted(int(t.get_text()) for t in ax.texts)
    plt.close(fig)
    assert labels == [1, 2, 3, 4, 5, 6, 7]
