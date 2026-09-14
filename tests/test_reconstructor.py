import numpy as np

from tttrkit.ptuio.reconstructor import _build_line_windows


def test_build_line_windows_applies_delays_and_bidirectional_flags():
    start, stop, reversed_mask = _build_line_windows(
        np.array([100, 200, 300], dtype=np.uint64),
        line_duration=80,
        line_idx=np.array([0, 1, 2], dtype=np.int64),
        bidirectional=True,
        line_start_marker_delay=0.1,
        line_stop_marker_delay=-0.05,
        laser_sync_rate=100.0,
    )

    np.testing.assert_array_equal(start, [110, 210, 310])
    np.testing.assert_array_equal(stop, [175, 275, 375])
    np.testing.assert_array_equal(reversed_mask, [False, True, False])
