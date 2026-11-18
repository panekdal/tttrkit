"""estimate_bidirectional_prealign on synthetic probe chunks."""

import numpy as np
import pytest

from tttrkit.ptuio import utils
from tttrkit.ptuio.reconstructor import ScanConfig

from test_frame_sync import PERIOD, make_stream


def prealign(events, monkeypatch):
    # The probe chunk comes from the reader; hand over the events instead.
    # line_duration is set, so no pre-scan reads the (absent) reader.
    monkeypatch.setattr(
        utils, "_read_probe_chunk", lambda *args, **kwargs: (events, 0)
    )
    config = ScanConfig(
        lines=40, pixels=4, bidirectional=True, line_duration=PERIOD * 8 // 10
    )
    return utils.estimate_bidirectional_prealign(
        None, config, laser_sync_rate=40e6, verbose=False
    )


def test_line_without_stop_marker(monkeypatch):
    events = make_stream(frames=1, lines=40)
    stops = np.flatnonzero((events["special"] == 1) & (events["channel"] == 2))
    events = np.delete(events, stops[5])
    ds = prealign(events, monkeypatch)
    # 39 lines with a stop: each paired with its own interval
    assert ds.sizes["paired_interval"] == 39 - 1  # the last line has no next start
    np.testing.assert_array_equal(ds["periods_nsync"], PERIOD)
    np.testing.assert_array_equal(ds["durations_nsync"], PERIOD * 8 // 10)


def test_no_stop_markers(monkeypatch):
    events = make_stream(frames=1, lines=40)
    events = events[~((events["special"] == 1) & (events["channel"] == 2))]
    with pytest.raises(ValueError, match="needs line stop markers"):
        prealign(events, monkeypatch)
