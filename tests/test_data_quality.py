"""Data-quality accounting of ImageReconstructor: counts in Dataset.attrs,
DataQualityWarning only when losses are significant."""

import warnings

import numpy as np
import pytest

from tttrkit.ptuio.decoder import event_dtype
from tttrkit.ptuio.reconstructor import (
    DataQualityWarning,
    FrameSyncWarning,
    ImageReconstructor,
    ScanConfig,
)

from test_frame_sync import FRAMES, LINES, PERIOD, PIXELS, make_stream

IN_LINES = FRAMES * LINES  # one photon per line in make_stream


def _with(events, rows):
    extra = np.array(rows, dtype=event_dtype)
    out = np.concatenate([events, extra])
    return out[np.argsort(out["nsync"], kind="stable")]


def _photon_in_line(f, l, dtime=10, channel=0):
    """A photon in the middle of line l of frame f of make_stream()."""
    t = 5 * PERIOD + f * (LINES + 3) * PERIOD + l * PERIOD + PERIOD // 2
    return (t, dtime, channel, 0)


def run(events, chunk=None, tcspc_channels=1024, **cfg):
    kw = dict(lines=LINES, pixels=PIXELS, frames=FRAMES, max_detector=2)
    kw.update(cfg)
    rec = ImageReconstructor(
        ScanConfig(**kw), outputs=["photon_count"], tcspc_channels=tcspc_channels
    )
    step = chunk or len(events)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for a in range(0, len(events), step):
            rec.update(events[a : a + step])
        ds = rec.finalize()
    quality = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    return ds, quality


def test_frame_sync_warning_is_a_data_quality_warning():
    assert issubclass(FrameSyncWarning, DataQualityWarning)


@pytest.mark.parametrize("chunk", [None, 1, 7])
def test_clean_stream_attrs(chunk):
    ds, quality = run(make_stream(), chunk=chunk)
    assert quality == []
    assert ds.attrs == {
        "photons_in_lines": IN_LINES,
        "photons_discarded_tcspc": 0,
        "photons_discarded_detector": 0,
        "markers_total": FRAMES * (2 * LINES + 1),
        "markers_discarded": 0,
        "frame_markers_off_boundary": 0,
        "data_beyond_frames": 0,
    }


BIG = LINES * 200  # enough photons for one outlier to stay below 0.1 %


def test_single_tcspc_outlier_is_counted_not_warned():
    events = make_stream(lines=BIG)  # dtimes up to 100 * 2 + BIG - 1
    events = _with(events, [_photon_in_line(0, 3, dtime=5000)])
    ds, quality = run(events, lines=BIG, tcspc_channels=4096)
    assert ds.attrs["photons_discarded_tcspc"] == 1
    assert quality == []


def test_too_few_tcspc_channels_is_warned():
    # make_stream dtimes are 100 * frame + line: frame 2 has 200..207
    ds, quality = run(make_stream(), tcspc_channels=150)
    assert ds.attrs["photons_discarded_tcspc"] == LINES
    assert len(quality) == 1 and "tcspc_channels" in str(quality[0].message)


def test_stray_detector_channel_is_counted_not_warned():
    events = make_stream(lines=BIG)
    events = _with(events, [_photon_in_line(1, 5, channel=53)])
    ds, quality = run(events, lines=BIG, tcspc_channels=4096)
    assert ds.attrs["photons_discarded_detector"] == 1
    assert quality == []


def test_real_detector_beyond_max_detector_is_warned():
    events = make_stream()
    events = _with(events, [_photon_in_line(f, l, channel=3) for f in range(FRAMES) for l in range(LINES)])
    ds, quality = run(events)
    assert ds.attrs["photons_discarded_detector"] == IN_LINES
    assert len(quality) == 1 and "{3: 24}" in str(quality[0].message)


def test_markers_with_unconfigured_inputs():
    # one glitch (frame + line start + input 8) among thousands of markers
    events = make_stream(lines=BIG)
    ds, quality = run(
        _with(events, [(5 * PERIOD + 10, 0, 13, 1)]), lines=BIG, tcspc_channels=4096
    )
    assert ds.attrs["markers_discarded"] == 1
    assert quality == []

    # line stops wired to input 8 instead of 2: every stop is discarded
    events = make_stream()
    stops = (events["special"] == 1) & (events["channel"] == 2)
    events["channel"][stops] = 8
    ds, quality = run(events)
    assert ds.attrs["markers_discarded"] == FRAMES * LINES
    assert any("marker channels" in str(w.message) for w in quality)


def test_unestimated_line_duration_is_a_data_quality_warning():
    # a single line start marker: no line duration, nothing binned
    events = make_stream()
    line_markers = (events["special"] == 1) & np.isin(events["channel"], [1, 2])
    drop = np.flatnonzero(line_markers)[1:]
    ds, quality = run(np.delete(events, drop))
    assert any("could not be estimated" in str(w.message) for w in quality)
    assert ds.attrs["photons_in_lines"] == 0


def test_data_beyond_configured_frames_is_recorded():
    ds, quality = run(make_stream(frames=5))
    assert ds.attrs["data_beyond_frames"] == 1
    assert quality == []
