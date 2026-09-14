import numpy as np

from tttrkit.ptuio import analyze_stop_marker_timing
from tttrkit.ptuio.reconstructor import (
    ImageReconstructor,
    ScanConfig,
    SegmentReconstructor,
)


def test_analyze_stop_marker_timing_uses_all_markers_without_frame_markers():
    timing = analyze_stop_marker_timing(
        np.array([], dtype=np.uint64),
        np.array([100, 200, 300], dtype=np.uint64),
        np.array([180, 280], dtype=np.uint64),
    )

    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.intervals, [100, 100])
    np.testing.assert_array_equal(timing.durations, [80, 80])
    np.testing.assert_allclose(timing.phases, [0.8, 0.8])
    assert timing.median_interval == 100.0
    assert timing.median_duration == 80.0
    assert timing.median_phase == 0.8
    assert timing.window == (None, None)


def test_analyze_stop_marker_timing_uses_first_complete_frame_only():
    timing = analyze_stop_marker_timing(
        np.array([50, 350, 650], dtype=np.uint64),
        np.array([0, 100, 200, 300, 400, 500], dtype=np.uint64),
        np.array([80, 180, 280, 380, 480], dtype=np.uint64),
    )

    assert timing.window == (50, 350)
    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.intervals, [100, 100])
    np.testing.assert_array_equal(timing.durations, [80, 80])


def test_analyze_stop_marker_timing_uses_data_after_a_single_frame_marker():
    timing = analyze_stop_marker_timing(
        np.array([100], dtype=np.uint64),
        np.array([0, 100, 200, 300], dtype=np.uint64),
        np.array([80, 180, 280], dtype=np.uint64),
    )

    assert timing.window == (100, None)
    assert timing.pair_count == 2
    np.testing.assert_array_equal(timing.durations, [80, 80])


def test_reconstructors_use_frame_bounded_stop_marker_analysis():
    frame_nsyncs = np.array([50, 350], dtype=np.uint64)
    start_nsyncs = np.array([0, 100, 200, 300, 400], dtype=np.uint64)
    stop_nsyncs = np.array([80, 180, 280, 380], dtype=np.uint64)

    for reconstructor in (
        ImageReconstructor(ScanConfig()),
        SegmentReconstructor(ScanConfig()),
    ):
        reconstructor._compute_stop_phase(
            frame_nsyncs, start_nsyncs, stop_nsyncs
        )

        assert reconstructor.stop_marker_phase == 0.8
        assert reconstructor.line_duration == 80
