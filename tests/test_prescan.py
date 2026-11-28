"""First-frame pre-scan: ScanConfig checks and line duration estimate."""

import warnings

import numpy as np
import pytest

from tttrkit.ptuio.prescan import prescan_events
from tttrkit.ptuio.reconstructor import (
    ScanConfig,
    ScanConfigWarning,
    SegmentReconstructor,
)

from test_frame_sync import FRAMES, LINES, PERIOD, make_stream

DURATION = PERIOD * 8 // 10  # line stop at 0.8 period in make_stream


def scan(events, chunk=None, **cfg):
    """Pre-scan events cut into chunks; (FirstFrameScan, config, warnings)."""
    kw = dict(lines=LINES, pixels=4, frames=FRAMES, max_detector_channels=2)
    kw.update(cfg)
    config = ScanConfig(**kw)
    step = chunk or len(events)
    chunks = [events[a : a + step] for a in range(0, len(events), step)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = prescan_events(chunks, config)
    issued = [w for w in caught if issubclass(w.category, ScanConfigWarning)]
    return result, config, issued


def without_stops(events):
    stops = (events["special"] == 1) & (events["channel"] == 2)
    return events[~stops]


def with_frame_pauses(events, lines, pause):
    """Add a pause of `pause` sync units before every frame (time-lapse)."""
    frame = (events["nsync"].astype(np.int64) - 4 * PERIOD) // (
        (lines + 3) * PERIOD
    )
    out = events.copy()
    out["nsync"] += (np.maximum(frame, 0) * pause).astype(np.uint64)
    return out


def without_flyback_gap(events):
    """Move every line stop to the end of the line period; where it meets
    the next line start, merge both into one value-3 marker."""
    out = events.copy()
    stops = (out["special"] == 1) & (out["channel"] == 2)
    out["nsync"][stops] += np.uint64(PERIOD - DURATION)
    starts = (out["special"] == 1) & (out["channel"] == 1)
    merged = starts & np.isin(out["nsync"], out["nsync"][stops])
    out["channel"][merged] = 3
    absorbed = stops & np.isin(out["nsync"], out["nsync"][merged])
    out = out[~absorbed]
    return out[np.argsort(out["nsync"], kind="stable")]


@pytest.mark.parametrize("chunk", [None, 1, 7])
def test_stream_without_flyback_gap(chunk):
    events = without_flyback_gap(make_stream())
    assert np.count_nonzero(events["channel"][events["special"] == 1] == 3)
    result, config, issued = scan(events, chunk)
    assert result.line_duration == PERIOD
    assert result.line_duration_source == "stop_markers"
    assert len(result.issues) == 1 and len(issued) == 1


@pytest.mark.parametrize("frame_marker", ["start", "end"])
@pytest.mark.parametrize("chunk", [None, 1, 7])
def test_clean_stream(frame_marker, chunk):
    result, config, issued = scan(make_stream(frame_marker=frame_marker), chunk)
    assert issued == [] and result.issues == []
    assert result.line_duration == DURATION
    assert result.line_duration_source == "stop_markers"
    assert config.line_duration == DURATION  # filled in
    assert result.median_line_period == PERIOD
    assert result.measured_lines_per_frame == LINES


def test_result_does_not_depend_on_chunk_size():
    rng = np.random.default_rng(0)
    events = make_stream(lines=64)
    stops = (events["special"] == 1) & (events["channel"] == 2)
    events["nsync"][stops] += rng.integers(-60, 60, stops.sum()).astype(np.uint64)
    events = events[np.argsort(events["nsync"], kind="stable")]
    results = [scan(events, chunk, lines=64)[0] for chunk in (None, 1, 50, 333)]
    assert all(r == results[0] for r in results[1:])


@pytest.mark.parametrize("frame_marker", ["start", "end", None])
def test_frame_pauses_do_not_bias_the_estimate_without_stops(frame_marker):
    # Time-lapse with 2-line frames: half of all start-to-start intervals
    # hold a long pause, the estimate must use only those within a frame
    lines = 2
    events = with_frame_pauses(
        make_stream(frames=80, lines=lines, frame_marker=frame_marker),
        lines,
        pause=50 * PERIOD,
    )
    result, config, issued = scan(without_stops(events), lines=lines, frames=80)
    # The only issue: the line duration is a guess without stop markers
    assert len(issued) == 1 and "guessed" in str(issued[0].message)
    assert result.issues == [str(issued[0].message)]
    assert result.median_line_period == PERIOD
    assert result.line_duration_source == "line_period"
    assert result.line_duration == DURATION


def test_given_line_duration_is_kept_and_checked():
    result, config, issued = scan(make_stream(), line_duration=PERIOD)
    assert issued == []
    assert result.line_duration_source == "config"
    assert config.line_duration == PERIOD

    result, config, issued = scan(make_stream(), line_duration=PERIOD + 1)
    assert config.line_duration == PERIOD + 1
    assert len(issued) == 1 and "exceeds the line period" in str(issued[0].message)


@pytest.mark.parametrize("frame_marker", ["start", "end"])
@pytest.mark.parametrize("lines", [4, 16])
def test_wrong_lines_per_frame(frame_marker, lines):
    # The data has 8 lines per frame
    events = make_stream(frames=6, frame_marker=frame_marker)
    result, _, issued = scan(events, lines=lines, frames=6)
    assert result.measured_lines_per_frame == LINES
    assert len(issued) == 1 and "8 line starts apart" in str(issued[0].message)


def test_accumulations_count_in_lines_per_frame():
    # 8 line starts per frame = 4 lines x 2 accumulations
    _, _, issued = scan(make_stream(), lines=4, line_accumulations=(2,))
    assert issued == []


def test_data_not_beginning_with_a_frame():
    # The first 3 lines (and the first frame marker) are missing
    events = make_stream(frames=4)
    events = events[events["nsync"] >= 5 * PERIOD + 3 * PERIOD - PERIOD // 10]
    # frame markers after 5, 13, 21 line starts, none near 8
    _, _, issued = scan(events, frames=4)
    assert len(issued) == 1
    assert "No frame marker within 1 line(s) of line start 8" in str(
        issued[0].message
    )


def test_glitch_frame_marker_is_left_to_frame_sync():
    events = make_stream(extra_frame_marker=(0, 3))
    _, _, issued = scan(events)
    assert issued == []


def test_glitch_next_to_the_only_frame_marker_is_left_to_frame_sync():
    # Frame markers after 0 and 1 line starts: not frames 1 line long
    events = make_stream(frames=1, extra_frame_marker=(0, 1))
    result, _, issued = scan(events, frames=1)
    assert issued == [] and result.measured_lines_per_frame is None


def test_single_frame_end_marker_measures_the_frame():
    # Line 0 starts frame 0, so one end-of-frame marker spans a frame
    result, _, issued = scan(make_stream(frames=1, frame_marker="end"), frames=1)
    assert issued == [] and result.measured_lines_per_frame == LINES


def test_spurious_line_marker_before_the_first_frame_marker():
    # The frame marker comes one line late; the reconstruction resyncs on
    # it, so this is no configuration mismatch
    events = make_stream(frames=1)
    events = np.sort(
        np.concatenate(
            [events, np.array([(5 * PERIOD - PERIOD // 2, 0, 1, 1)], events.dtype)]
        ),
        order="nsync",
        kind="stable",
    )
    _, _, issued = scan(events, frames=1)
    assert issued == []


def test_dropped_line_marker_in_the_first_frame_is_left_to_frame_sync():
    # 39 line starts in frame 0, 40 configured: within the tolerance (4)
    lines = 40
    events = make_stream(frames=6, lines=lines, drop_line=(0, 5))
    _, _, issued = scan(events, lines=lines, frames=6)
    assert issued == []


def test_without_frame_markers():
    # more lines than the scan reads: nothing to check the frames against
    result, _, issued = scan(make_stream(frames=20, frame_marker=None), frames=20)
    assert issued == [] and result.measured_lines_per_frame is None
    assert result.line_duration == DURATION


def test_without_frame_markers_whole_frames_by_line_count():
    # the scan reaches the end of the data: 3 frames of 8 lines
    _, _, issued = scan(make_stream(frame_marker=None))
    assert issued == []


def test_frame_sync_count_ignores_frame_markers():
    result, _, issued = scan(
        make_stream(frames=20), lines=4, frames=20, frame_sync="count"
    )
    assert issued == [] and result.measured_lines_per_frame is None


def test_fewer_lines_than_one_frame():
    _, config, issued = scan(make_stream(frames=1), lines=16)
    assert len(issued) == 1
    assert "holds 8 line start markers, fewer than one frame" in str(
        issued[0].message
    )
    # the line duration is still estimated from what there is
    assert config.line_duration == DURATION


def test_single_frame_without_trailing_marker_is_ok():
    _, _, issued = scan(make_stream(frames=1), frames=1)
    assert issued == []


def test_reads_only_the_first_frame():
    # 8-line frames: 128 lines (16 frames) + tolerance (1) + 1 line starts
    events = make_stream(frames=40)
    consumed = []

    def chunks():
        for a in range(0, len(events), 10):
            consumed.append(a)
            yield events[a : a + 10]

    config = ScanConfig(lines=LINES, pixels=4, frames=40)
    result = prescan_events(chunks(), config)
    assert result.issues == []
    assert len(consumed) * 10 < len(events) / 2


def test_segment_reconstructor_checks_a_given_line_duration():
    events = make_stream()
    with pytest.warns(ScanConfigWarning, match="exceeds the line period"):
        SegmentReconstructor(
            ScanConfig(lines=LINES, pixels=4, line_duration=PERIOD + 1)
        ).reconstruct(events)


def test_segment_reconstructor_estimate_excludes_frame_pauses():
    lines = 2
    events = with_frame_pauses(
        make_stream(frames=20, lines=lines), lines, pause=50 * PERIOD
    )
    config = ScanConfig(lines=lines, pixels=4)
    with pytest.warns(ScanConfigWarning, match="guessed"):
        ds = SegmentReconstructor(config).reconstruct(without_stops(events))
    assert ds["line_duration"].item() == DURATION
    assert config.line_duration is None  # the caller's config is unchanged
