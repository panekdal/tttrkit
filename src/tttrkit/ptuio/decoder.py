"""Decoding of 32-bit T3 records.

The record layout fixes several numbers that the rest of tttrkit uses; they
are defined here, once:

    bit  0-9   nsync    sync counter, wraps every T3_WRAPAROUND syncs
    bit 10-24  dtime    TCSPC channel, DTIME_CHANNELS values
    bit 25-30  channel  detector (photons) or marker value (special records)
    bit 31     special  1 for markers and overflows
"""

import numpy as np
from enum import IntEnum


class TTTRType(IntEnum):
    """TTResultFormat_TTTRRecType of the T3 formats decode_t3 reads."""

    HydraHarpT3 = 0x00010304
    HydraHarp2T3 = 0x01010304
    TimeHarp260NT3 = 0x00010305
    TimeHarp260PT3 = 0x00010306
    GenericT3 = 0x00010307  # MultiHarp, HydraHarp 500, PicoHarp 330


T3_BITS_PER_RECORD = 32
T3_NSYNC_MASK = 0x3FF
# Syncs per overflow period; fixed by the 10-bit nsync field
T3_WRAPAROUND = T3_NSYNC_MASK + 1
T3_DTIME_MASK = 0x7FFF
# Number of TCSPC channels a 15-bit dtime can address
DTIME_CHANNELS = T3_DTIME_MASK + 1
T3_CHANNEL_MASK = 0x3F
# Number of detector channels a 6-bit channel field can address (0-63)
DETECTOR_CHANNELS = T3_CHANNEL_MASK + 1
# Special record with this channel value: a sync counter overflow (a photon
# record with it is detector 63)
OVERFLOW_CHANNEL = T3_CHANNEL_MASK
# Special records with channel 1-15 are markers; each of the four bits is
# one physical marker input, so coinciding markers combine their bits
MARKER_BITS = (1, 2, 4, 8)
MARKER_VALUE_LIMIT = 16


def check_marker_bit(name: str, value) -> int:
    """``value`` as an int if it is one marker input bit, else raise."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer marker bit")
    if value not in MARKER_BITS:
        raise ValueError(f"{name} must be one of {MARKER_BITS}, got {value}")
    return int(value)


event_dtype = np.dtype(
    [
        ("nsync", np.uint64),
        ("dtime", np.uint16),
        ("channel", np.uint8),
        ("special", np.uint8),
    ]
)


def photon_events(events: np.ndarray) -> np.ndarray:
    """All photon events (non-special records), on any of the
    DETECTOR_CHANNELS channels. Channel 63 is an overflow only in special
    records; in a photon record it is the 64th detector."""
    return events[events["special"] == 0]


def resolve_markers(
    events: np.ndarray,
    frame_start_marker_channel: int,
    line_start_marker_channel: int,
    line_stop_marker_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract compatible frame, line-start, and line-stop marker events.

    Marker channels are bitmasks, so a single event may represent multiple
    physical marker inputs. Frame/start and frame/stop coincidences are valid.
    A line start coinciding with a line stop (e.g. value 3) is a line start:
    the stop ends the previous line, as on scanners without a flyback gap
    between lines. It is therefore not returned as a line stop.
    Markers that include an unconfigured input are discarded as glitches.
    """
    check_marker_bit("frame_start_marker_channel", frame_start_marker_channel)
    check_marker_bit("line_start_marker_channel", line_start_marker_channel)
    check_marker_bit("line_stop_marker_channel", line_stop_marker_channel)

    configured_bits = (
        frame_start_marker_channel
        | line_start_marker_channel
        | line_stop_marker_channel
    )
    is_marker = (
        (events["special"] != 0)
        & (events["channel"] < MARKER_VALUE_LIMIT)
        & ((events["channel"] & configured_bits) == events["channel"])
    )
    # this discards markers that coincide with unconfigured
    # marker channels as these usually are glitches

    markers = events[is_marker]
    channels = markers["channel"]

    has_frame = (channels & frame_start_marker_channel) != 0
    has_line_start = (channels & line_start_marker_channel) != 0
    has_line_stop = (channels & line_stop_marker_channel) != 0

    return (
        markers[has_frame],
        markers[has_line_start],
        markers[has_line_stop & ~has_line_start],
    )


def marker_events(events: np.ndarray) -> np.ndarray:
    """All marker events (special records other than overflows)."""
    return events[(events["channel"] < OVERFLOW_CHANNEL) & (events["special"] != 0)]


def decode_t3(records):
    out = np.empty(records.shape[0], dtype=event_dtype)

    out["nsync"] = (records & T3_NSYNC_MASK).astype(np.uint64)  # first 10 bits
    out["dtime"] = ((records >> 10) & T3_DTIME_MASK).astype(np.uint16)  # next 15 bits
    out["channel"] = ((records >> 25) & T3_CHANNEL_MASK).astype(np.uint8)  # next 6 bits
    out["special"] = ((records >> 31) & 0x1).astype(np.uint8)  # last one bit

    return out


class T3OverflowCorrector:
    """Decode T3 records and add the overflow periods to nsync.

    Stateful: feed the chunks of one file in order, from the start of the
    data. The wraparound is T3_WRAPAROUND, fixed by the record format.
    """

    def __init__(self):
        self.wraparound = T3_WRAPAROUND
        self.overflow_carry = 0

    def correct(self, records: np.ndarray) -> np.ndarray:
        decoded = decode_t3(records)

        is_overflow = (
            (decoded["special"] == 1)
            & (decoded["channel"] == OVERFLOW_CHANNEL)
        )

        # In Generic T3, nsync=0 encodes one overflow period.
        overflow_count = np.where(
            is_overflow,
            np.where(decoded["nsync"] == 0, 1, decoded["nsync"]),
            0,
        )

        overflow_correction = (
            np.cumsum(overflow_count, dtype=np.uint64) * self.wraparound
        )
        overflow_total = overflow_correction + self.overflow_carry

        decoded["nsync"] += overflow_total

        if len(overflow_total):
            self.overflow_carry = overflow_total[-1]

        return decoded
