import numpy as np
from enum import IntEnum

class TTTRType(IntEnum):
    HydraHarp2T3 = 0x00010304
    MultiHarpT3 = 0x00010305
    TimeHarp260NT3 = 0x00010306
    GenericT3 = 0x00010307  # fallback


event_dtype = np.dtype(
    [
        ("nsync", np.uint64),
        ("dtime", np.uint16),
        ("channel", np.uint8),
        ("special", np.uint8),
    ]
)


def get_photons(events: np.ndarray):
    return events[(events["channel"] < 63) & (events["special"] == 0)]


def resolve_markers(
    events: np.ndarray,
    frame_marker_mask: int,
    line_start_marker_mask: int,
    line_stop_marker_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract compatible frame, line-start, and line-stop marker events.

    Marker channels are bitmasks, so a single event may represent multiple
    physical marker inputs. Frame/start and frame/stop coincidences are valid,
    but one event cannot be both a line start and a line stop. Such events are
    excluded from every returned role.
    """
    marker_masks = (
        frame_marker_mask,
        line_start_marker_mask,
        line_stop_marker_mask,
    )
    if any(
        not isinstance(mask, (int, np.integer)) or mask not in (1, 2, 4, 8)
        for mask in marker_masks
    ):
        raise ValueError("marker masks must be one of 1, 2, 4, or 8")

    is_marker = (events["special"] != 0) & (events["channel"] < 16)
    marker_events = events[is_marker]
    channels = marker_events["channel"]

    has_frame = (channels & frame_marker_mask) != 0
    has_line_start = (channels & line_start_marker_mask) != 0
    has_line_stop = (channels & line_stop_marker_mask) != 0

    compatible = ~(has_line_start & has_line_stop)

    return (
        marker_events[compatible & has_frame],
        marker_events[compatible & has_line_start],
        marker_events[compatible & has_line_stop],
    )


def get_markers(events: np.ndarray, marker_mask: int) -> np.ndarray:
    """Return marker events containing a configured physical marker input.

    ``marker_mask`` is the bit value of one marker input (1, 2, 4, or 8).
    A composite marker is retained whenever it contains that input; for
    example, mask ``4`` matches both marker value ``4`` and value ``6``.

    This helper is intended for analysis. Reconstruction uses
    :func:`resolve_markers` instead, because it validates marker roles across
    the same event and rejects contradictory line-start/line-stop coincidences.
    """
    if not isinstance(marker_mask, (int, np.integer)) or marker_mask <= 0:
        raise ValueError("marker_mask must be a positive integer")

    is_marker = (events["special"] != 0) & (events["channel"] < 63)
    contains_marker = (events["channel"] & marker_mask) != 0
    return events[is_marker & contains_marker]


def decode_t3(records):
    out = np.empty(
        records.shape[0],
        dtype=[
            ("nsync", np.uint64),
            ("dtime", np.uint16),
            ("channel", np.uint8),
            ("special", np.uint8),
        ],
    )

    out["nsync"] = (records & 0x3FF).astype(np.uint64)  # first 10 bits
    out["dtime"] = ((records >> 10) & 0x7FFF).astype(np.uint16)  # next 15 bits
    out["channel"] = ((records >> 25) & 0x3F).astype(np.uint8)  # next 6 bits
    out["special"] = ((records >> 31) & 0x1).astype(np.uint8)  # last one bit

    return out

class T3OverflowCorrector:
    def __init__(self, wraparound=1024):
        self.wraparound = wraparound
        self.overflow_carry = 0  # Total number of wraparound events so far

    def correct(self, records: np.ndarray) -> np.ndarray:
        decoded = decode_t3(records)

        is_overflow = (decoded["special"] == 1) & (decoded["channel"] == 0x3F)
        overflow_correction = (
            np.cumsum(is_overflow * decoded["nsync"]) * self.wraparound
        )
        overflow_total = overflow_correction + self.overflow_carry

        # Apply correction to *all* events
        decoded["nsync"] += overflow_total

        # Update carry for next chunk
        if len(overflow_correction) > 0:
            self.overflow_carry = overflow_total[-1]

        return decoded
