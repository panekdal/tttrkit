import numpy as np

from tttrkit.ptuio.marker_timing import (
    analyze_stop_marker_timing,
    coincident_stop_issue,
    estimate_line_duration,
)


def test_analyze_stop_marker_timing_uses_all_markers_without_frame_markers():
    timing = analyze_stop_marker_timing(
        np.array([], dtype=np.uint64),
        np.array([100, 200, 300], dtype=np.uint64),
        np.array([180, 280], dtype=np.uint64),
    )

    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.paired_intervals, [100, 100])
    np.testing.assert_array_equal(timing.durations, [80, 80])
    np.testing.assert_allclose(timing.phases, [0.8, 0.8])
    assert timing.median_duration == 80.0
    assert timing.median_phase == 0.8


def test_analyze_stop_marker_timing_uses_first_complete_frame_only():
    timing = analyze_stop_marker_timing(
        np.array([50, 350, 650], dtype=np.uint64),
        np.array([0, 100, 200, 300, 400, 500], dtype=np.uint64),
        np.array([80, 180, 280, 380, 480], dtype=np.uint64),
    )

    # line starts 100, 200, 300 within [50, 350)
    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.paired_intervals, [100, 100])
    np.testing.assert_array_equal(timing.durations, [80, 80])


def test_stop_at_next_start_ends_the_line_without_flyback_gap():
    # value-3 markers: each stop coincides with the next line's start
    starts = np.array([100, 200, 300, 400], dtype=np.uint64)
    stops = np.array([200, 300, 400, 500], dtype=np.uint64)

    timing = analyze_stop_marker_timing(
        np.array([], dtype=np.uint64), starts, stops
    )
    assert timing.pair_count == 3
    np.testing.assert_array_equal(timing.durations, [100, 100, 100])
    assert timing.median_phase == 1.0

    estimate = estimate_line_duration(
        np.array([], dtype=np.uint64), starts, stops
    )
    assert (estimate.duration, estimate.source) == (100, "stop_markers")

    issue = coincident_stop_issue(starts, stops)
    assert issue.startswith("3 of 4 line stop markers")


def test_accidental_coincidence_is_reported_but_does_not_move_the_median():
    starts = np.array([100, 200, 300, 400, 500], dtype=np.uint64)
    stops = np.array([180, 280, 400, 480], dtype=np.uint64)  # 3rd: gap < 1 sync

    estimate = estimate_line_duration(
        np.array([], dtype=np.uint64), starts, stops
    )
    assert estimate.duration == 80

    issue = coincident_stop_issue(starts, stops)
    assert issue.startswith("1 of 4 line stop markers")
    assert coincident_stop_issue(starts, np.array([180, 280], dtype=np.uint64)) is None


def test_analyze_stop_marker_timing_uses_data_after_a_single_frame_marker():
    timing = analyze_stop_marker_timing(
        np.array([100], dtype=np.uint64),
        np.array([0, 100, 200, 300], dtype=np.uint64),
        np.array([80, 180, 280], dtype=np.uint64),
    )

    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.durations, [80, 80])
