import numpy as np

from tttrkit.ptuio.decoder import event_dtype, resolve_markers


def make_markers(channels):
    events = np.zeros(len(channels), dtype=event_dtype)
    events["nsync"] = np.arange(len(channels))
    events["channel"] = channels
    events["special"] = 1
    return events


def test_resolve_markers_allows_frame_coincidences_and_rejects_start_stop():
    events = make_markers([5, 6, 7, 1, 2, 4, 63])

    frame, starts, stops = resolve_markers(events, 4, 1, 2)

    assert frame["channel"].tolist() == [5, 6, 4]
    assert starts["channel"].tolist() == [5, 1]
    assert stops["channel"].tolist() == [6, 2]


def test_resolve_markers_rejects_non_marker_events():
    events = make_markers([1, 2])
    events[0]["special"] = 0

    frame, starts, stops = resolve_markers(events, 4, 1, 2)

    assert len(frame) == 0
    assert len(starts) == 0
    assert stops["channel"].tolist() == [2]