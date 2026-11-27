"""Acquisition values come from the file header, never from assumed defaults."""

import struct

import numpy as np
import pytest

from tttrkit.ptuio.calibration import estimate_tcspc_channels
from tttrkit.ptuio.decoder import DTIME_CHANNELS, T3_WRAPAROUND, TTTRType, event_dtype
from tttrkit.ptuio.file import Header, TagTypes
from tttrkit.ptuio.reader import TTTRReader
from tttrkit.ptuio.reconstructor import TraceReconstructor

T3_TAGS = {
    "TTResultFormat_TTTRRecType": (TagTypes.tyInt8, TTTRType.HydraHarp2T3),
    "TTResultFormat_BitsPerRecord": (TagTypes.tyInt8, 32),
    "TTResult_SyncRate": (TagTypes.tyInt8, 80_000_000),
    "MeasDesc_Resolution": (TagTypes.tyFloat8, 32e-12),
}


def write_ptu(path, tags):
    """A minimal .ptu file: the given tags (name: (type, value)), no records."""
    with open(path, "wb") as f:
        f.write(b"PQTTTR\0\0" + b"1.0\0\0\0\0\0")
        for name, (typecode, value) in tags.items():
            f.write(struct.pack("<32siI", name.encode(), -1, typecode))
            fmt = "d" if typecode == TagTypes.tyFloat8 else "q"
            f.write(struct.pack(fmt, value))
        f.write(struct.pack("<32siI", b"Header_End", -1, TagTypes.tyEmpty8))
        f.write(struct.pack("q", 0))
    return str(path)


def test_acquisition_values_from_the_header(tmp_path):
    header = TTTRReader(write_ptu(tmp_path / "ok.ptu", T3_TAGS)).header
    assert header.sync_rate == 80e6
    assert header.tcspc_resolution == 32e-12


@pytest.mark.parametrize("tag", ["TTResult_SyncRate", "MeasDesc_Resolution"])
def test_missing_acquisition_tag_raises(tmp_path, tag):
    tags = {k: v for k, v in T3_TAGS.items() if k != tag}
    header = Header(write_ptu(tmp_path / "missing.ptu", tags))
    with pytest.raises(ValueError, match=tag):
        getattr(header, "sync_rate" if tag == "TTResult_SyncRate" else "tcspc_resolution")


@pytest.mark.parametrize(
    "tag, value, match",
    [
        ("TTResultFormat_TTTRRecType", 0x00010303, "unsupported record type"),  # PicoHarp T3
        ("TTResultFormat_TTTRRecType", 0x00010207, "unsupported record type"),  # Generic T2
        ("TTResultFormat_BitsPerRecord", 64, "bits per record"),
        ("TTResultFormat_WrapAround", 65536, "wrap every 1024"),
    ],
)
def test_reader_refuses_records_it_cannot_decode(tmp_path, tag, value, match):
    tags = {**T3_TAGS, tag: (TagTypes.tyInt8, value)}
    path = write_ptu(tmp_path / "bad.ptu", tags)
    Header(path)  # the header itself can still be read
    with pytest.raises(ValueError, match=match):
        TTTRReader(path)


def test_wraparound_tag_is_optional_and_checked(tmp_path):
    tags = {**T3_TAGS, "TTResultFormat_WrapAround": (TagTypes.tyInt8, T3_WRAPAROUND)}
    TTTRReader(write_ptu(tmp_path / "wrap.ptu", tags))


def test_estimate_tcspc_channels():
    # 80 MHz at 32 ps: 391 channels per period
    assert estimate_tcspc_channels(80e6, 32e-12, buffer=10) == 401
    # capped at what a 15-bit dtime can address
    assert estimate_tcspc_channels(1e6, 1e-12) == DTIME_CHANNELS


def test_trace_reconstructor_uses_the_given_marker_channels():
    events = np.zeros(3, dtype=event_dtype)
    events["nsync"] = [10, 20, 30]
    events["special"] = 1
    events["channel"] = [8, 2, 1]  # frame, line start, line stop (rewired)
    rec = TraceReconstructor(
        sync_rate=10.0,
        stop_time_s=10.0,
        outputs=["markers"],
        frame_start_marker_channel=8,
        line_start_marker_channel=2,
        line_stop_marker_channel=1,
    )
    rec.update(events)
    ds = rec.finalize()
    assert ds["frame_start_marker_times_s"].values.tolist() == [1.0]
    assert ds["line_start_marker_times_s"].values.tolist() == [2.0]
    assert ds["line_stop_marker_times_s"].values.tolist() == [3.0]


def test_trace_reconstructor_requires_sync_rate_and_window():
    with pytest.raises(TypeError):
        TraceReconstructor(stop_time_s=1.0)
    with pytest.raises(TypeError):
        TraceReconstructor(sync_rate=40e6)
