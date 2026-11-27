"""Frame assignment in ImageReconstructor: line counting and frame_sync modes.

Synthetic streams carry one photon per line, in the middle of the line, with
dtime = 100 * true_frame + true_line, so every binned photon reveals where it
should have gone.
"""

import warnings

import numpy as np
import pytest

from tttrkit.ptuio.decoder import event_dtype
from tttrkit.ptuio.prescan import prescan_events
from tttrkit.ptuio.reconstructor import (
    FrameSyncWarning,
    ImageReconstructor,
    ScanConfig,
)

PERIOD = 1000  # line period in sync units
LINES = 8
PIXELS = 4
FRAMES = 3
# Acquisition values ImageReconstructor requires (from the header for files)
ACQUISITION = dict(sync_rate=40e6, tcspc_resolution=5e-12)


def make_stream(
    frames=FRAMES,
    lines=LINES,
    frame_marker="start",
    drop_line=None,
    extra_line=None,
    extra_frame_marker=None,
):
    """Return a time-sorted event array.

    frame_marker: "start" (before line 0, after an inter-frame gap),
        "end" (after the last line, as on Leica SP8) or None.
    drop_line / extra_line: (frame, line) whose line start marker is removed /
        duplicated by a spurious marker a quarter period later.
    extra_frame_marker: (frame, line) before which a spurious frame marker
        is inserted.
    """
    ev = []

    def add(nsync, channel, special, dtime=0):
        ev.append((nsync, dtime, channel, special))

    t = 5 * PERIOD
    for f in range(frames):
        if frame_marker == "start":
            add(t - PERIOD // 4, 4, 1)
        for l in range(lines):
            if extra_frame_marker == (f, l):
                add(t - PERIOD // 10, 4, 1)
            if drop_line != (f, l):
                add(t, 1, 1)
            if extra_line == (f, l):
                add(t + PERIOD // 4, 1, 1)
            add(t + PERIOD * 8 // 10, 2, 1)  # line stop
            add(t + PERIOD * 4 // 10, 0, 0, dtime=100 * f + l)  # photon
            t += PERIOD
        if frame_marker == "end":
            add(t - PERIOD // 2, 4, 1)
        t += 3 * PERIOD  # inter-frame gap
    add(t + 10 * PERIOD, 0, 0, dtime=999)  # trailing photon, belongs nowhere
    out = np.array(ev, dtype=event_dtype)
    return out[np.argsort(out["nsync"], kind="stable")]


def reconstruct(events, chunk=None, **cfg):
    kw = dict(lines=LINES, pixels=PIXELS, frames=FRAMES, max_detector_channels=2)
    kw.update(cfg)
    config = ScanConfig(**kw)
    bounds = list(range(0, len(events), chunk or len(events))) + [len(events)]
    chunks = [events[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prescan_events(chunks, config)
        rec = ImageReconstructor(
            config, outputs=["mean_arrival_time"], tcspc_channels=1024,
            **ACQUISITION,
        )
        for c in chunks:
            rec.update(c)
        rec.finalize()
    sync_warnings = [w for w in caught if issubclass(w.category, FrameSyncWarning)]
    return rec, sync_warnings


def placed(rec):
    """(frame, line) -> dtime of the photon binned there (-1 if empty)."""
    count = rec.photon_count[:, :, PIXELS // 2, 0]
    assert count.max() <= 1, "two photons binned into one pixel"
    return np.where(count == 1, rec.arrival_sum[:, :, PIXELS // 2, 0], -1)


def expected(frames=FRAMES, lines=LINES):
    f, l = np.meshgrid(np.arange(frames), np.arange(lines), indexing="ij")
    return 100 * f + l


@pytest.mark.parametrize("mode", ["count", "validate", "resync"])
@pytest.mark.parametrize("frame_marker", ["start", "end"])
@pytest.mark.parametrize("chunk", [None, 1, 7])
def test_clean_stream(mode, frame_marker, chunk):
    rec, warns = reconstruct(
        make_stream(frame_marker=frame_marker), chunk=chunk, frame_sync=mode
    )
    np.testing.assert_array_equal(placed(rec), expected())
    assert warns == []
    if mode != "count":
        assert [e["action"] for e in rec.frame_sync_events] == ["ok"] * FRAMES


def test_no_frame_markers():
    events = make_stream(frame_marker=None)
    rec, warns = reconstruct(events, frame_sync="count")
    np.testing.assert_array_equal(placed(rec), expected())
    assert warns == []

    rec, warns = reconstruct(events, frame_sync="validate")
    np.testing.assert_array_equal(placed(rec), expected())
    assert any("No frame markers" in str(w.message) for w in warns)


@pytest.mark.parametrize("frame_marker", ["start", "end"])
def test_dropped_line_marker(frame_marker):
    events = make_stream(frame_marker=frame_marker, drop_line=(0, 3))

    # count / validate: from the drop on, line k holds true line k + 1
    rec, warns = reconstruct(events, chunk=5, frame_sync="validate")
    got = placed(rec)
    assert got[1, 7] == expected()[2, 0]
    assert got[2, 0] == expected()[2, 1]
    assert len(warns) >= 1

    # resync: damage confined to frame 0, later frames intact
    rec, warns = reconstruct(events, chunk=5, frame_sync="resync")
    got = placed(rec)
    np.testing.assert_array_equal(got[1:], expected()[1:])
    np.testing.assert_array_equal(got[0, :3], expected()[0, :3])
    actions = [e["action"] for e in rec.frame_sync_events]
    assert "resync" in actions


def test_spurious_line_marker_resync():
    events = make_stream(extra_line=(0, 5))
    rec, _ = reconstruct(events, chunk=3, frame_sync="resync")
    got = placed(rec)
    np.testing.assert_array_equal(got[1:], expected()[1:])


def test_spurious_frame_marker_is_ignored():
    # Mid-frame marker: 4 of 8 lines off, beyond the 10 % tolerance
    events = make_stream(extra_frame_marker=(1, 4))
    for mode in ("validate", "resync"):
        rec, warns = reconstruct(events, chunk=4, frame_sync=mode)
        np.testing.assert_array_equal(placed(rec), expected())
        assert len(warns) == 1
        bad = [e for e in rec.frame_sync_events if e["action"] != "ok"]
        assert [e["action"] for e in bad] == [
            "ignored" if mode == "resync" else "reported"
        ]


def test_wrong_lines_config_is_reported():
    # Hardware scans 8 lines/frame, config says 6
    rec, warns = reconstruct(make_stream(), lines=6, frame_sync="validate")
    assert any("off the boundary" in str(w.message) for w in warns)
    assert rec.photon_count.shape[1] == 6


def test_too_few_lines_is_reported():
    rec, warns = reconstruct(make_stream(frames=2), frame_sync="count")
    np.testing.assert_array_equal(placed(rec)[:2], expected(frames=2))
    assert (placed(rec)[2] == -1).all()
    assert any("Only 16 of the 24" in str(w.message) for w in warns)


def test_lines_beyond_configured_frames_are_dropped():
    rec, warns = reconstruct(make_stream(frames=5), frame_sync="validate")
    np.testing.assert_array_equal(placed(rec), expected())
    assert warns == []


def test_default_mode_is_resync():
    geometry = dict(lines=LINES, pixels=PIXELS)
    assert ScanConfig(**geometry).frame_sync == "resync"
    assert ScanConfig.from_dict(geometry).frame_sync == "resync"
    with pytest.raises(ValueError):
        ScanConfig(**geometry, frame_sync="sometimes")


def _add_line_marker(events, nsync):
    extra = np.array([(nsync, 0, 1, 1)], dtype=event_dtype)
    out = np.concatenate([events, extra])
    return out[np.argsort(out["nsync"], kind="stable")]


@pytest.mark.parametrize("chunk", [None, 1, 3, 7])
def test_several_spurious_lines_discarded_at_any_chunk_size(chunk):
    # Two spurious line markers in frame 0 of 40 lines (tolerance 4 lines):
    # both excess lines must be discarded before they are binned
    lines = 40
    events = make_stream(lines=lines, extra_line=(0, 30))
    events = _add_line_marker(events, 5 * PERIOD + 33 * PERIOD + PERIOD // 4)
    rec, _ = reconstruct(events, chunk=chunk, lines=lines, frame_sync="resync")
    assert [e["action"] for e in rec.frame_sync_events] == ["ok", "resync", "ok"]
    np.testing.assert_array_equal(placed(rec)[1:], expected(lines=lines)[1:])


def test_line_duration_estimate_does_not_depend_on_chunk_size():
    # More lines than LINE_DURATION_ESTIMATE_LINES, with jittered line stops
    rng = np.random.default_rng(0)
    events = make_stream(lines=64)
    stops = (events["special"] == 1) & (events["channel"] == 2)
    events["nsync"][stops] += rng.integers(-60, 60, stops.sum()).astype(np.uint64)
    events = events[np.argsort(events["nsync"], kind="stable")]
    durations, counts = set(), []
    for chunk in (None, 1, 50, 333):
        rec, _ = reconstruct(events, chunk=chunk, lines=64)
        durations.add(rec.line_duration)
        assert rec.config.line_duration == rec.line_duration
        counts.append(rec.photon_count.copy())
    assert len(durations) == 1
    for c in counts[1:]:
        np.testing.assert_array_equal(c, counts[0])


@pytest.mark.parametrize("chunk", [None, 1, 4, 9])
def test_negative_start_delay_keeps_photons_before_marker(chunk):
    # A photon 50 sync units before each line marker belongs to that line
    # when the start delay is -100
    events = make_stream()
    starts = events[(events["special"] == 1) & (events["channel"] == 1)]
    early = starts.copy()
    early["nsync"] -= 50
    early["special"] = 0
    early["channel"] = 1  # second detector
    events = np.concatenate([events, early])
    events = events[np.argsort(events["nsync"], kind="stable")]
    rec, _ = reconstruct(
        events, chunk=chunk, line_start_marker_delay=-100,
        line_stop_marker_delay=-100,
    )
    assert rec.photon_count[..., 1].sum() == FRAMES * LINES
    assert (rec.photon_count[:, :, 0, 1] == 1).all()
