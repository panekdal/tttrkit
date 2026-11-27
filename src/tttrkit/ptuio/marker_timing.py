"""Frame and line timing from marker arrays (nsync units); no I/O, no config."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Without line stop markers the line duration is guessed as this fraction of
# the line period (the remainder being the flyback). A guess, so it is
# reported with a ScanConfigWarning.
GUESSED_LINE_PHASE = 0.80


def frame_boundary_mismatch(lines_before: int, lines_per_frame: int):
    """(frame, mismatch) of a frame marker preceded by ``lines_before`` lines.

    ``frame`` is the nearest frame boundary and ``mismatch`` the number of
    lines the marker is off it (0 for a marker on a boundary). Shared by the
    frame sync of ImageReconstructor and the first-frame pre-scan.
    """
    frame = int(np.floor(lines_before / lines_per_frame + 0.5))
    return frame, lines_before - frame * lines_per_frame


@dataclass(frozen=True)
class StopMarkerTiming:
    """Start-to-stop timing of the lines that have a stop marker.

    ``durations`` (start to stop) and ``paired_intervals`` (start to next
    start) are per paired line; ``phases`` is their ratio.
    """

    pair_count: int
    phases: NDArray[np.float64]
    durations: NDArray[np.int64]
    paired_intervals: NDArray[np.int64]
    median_phase: float | None
    median_duration: float | None


def _as_nsyncs(name: str, markers: NDArray[np.uint64]) -> NDArray[np.int64]:
    nsyncs = np.asarray(markers)
    if nsyncs.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.issubdtype(nsyncs.dtype, np.integer):
        raise TypeError(f"{name} must contain integer nsync values")

    nsyncs = nsyncs.astype(np.int64, copy=False)
    if np.any(nsyncs[1:] < nsyncs[:-1]):
        raise ValueError(f"{name} must be sorted in ascending nsync order")
    return nsyncs


def analyze_stop_marker_timing(
    frame_nsyncs: NDArray[np.uint64],
    start_nsyncs: NDArray[np.uint64],
    stop_nsyncs: NDArray[np.uint64],
) -> StopMarkerTiming:
    """Analyze line timing within the first complete frame when available.

    With two or more frame markers, only markers in ``[frame[0], frame[1])``
    are used. With one frame marker, only markers at or after that marker are
    used. With no frame markers, all supplied line markers are used.
    """
    frame = _as_nsyncs("frame_nsyncs", frame_nsyncs)
    starts = _as_nsyncs("start_nsyncs", start_nsyncs)
    stops = _as_nsyncs("stop_nsyncs", stop_nsyncs)

    if len(frame):
        starts = starts[starts >= frame[0]]
        stops = stops[stops >= frame[0]]
    if len(frame) >= 2:
        starts = starts[starts < frame[1]]
        stops = stops[stops < frame[1]]

    intervals = np.diff(starts)
    # Each line can use only a stop after its start and before its next start.
    stop_indices = np.searchsorted(stops, starts[:-1], side="right")
    paired = stop_indices < len(stops)
    paired[paired] &= stops[stop_indices[paired]] < starts[1:][paired]

    durations = stops[stop_indices[paired]] - starts[:-1][paired]
    paired_intervals = intervals[paired]
    valid = (
        (durations > 0)
        & (paired_intervals > 0)
        & (durations < paired_intervals)
    )
    durations = durations[valid]
    paired_intervals = paired_intervals[valid]
    phases = durations / paired_intervals

    return StopMarkerTiming(
        pair_count=len(durations),
        phases=phases,
        durations=durations,
        paired_intervals=paired_intervals,
        median_phase=float(np.median(phases)) if len(phases) else None,
        median_duration=float(np.median(durations)) if len(durations) else None,
    )


@dataclass(frozen=True)
class LineDurationEstimate:
    """Line duration estimated from line markers (all values in nsync units).

    ``source`` is "stop_markers" (median start-to-stop time), "line_period"
    (GUESSED_LINE_PHASE times the median line period, without stop markers)
    or None when nothing could be estimated. ``median_period`` is the median
    start-to-start interval of lines followed by a line of the same frame,
    None if there is none.
    """

    duration: int | None
    source: str | None
    median_period: float | None


def within_frame_intervals(
    start_nsyncs: NDArray[np.uint64],
    frame_nsyncs: NDArray[np.uint64],
    lines_per_frame: int | None = None,
    first_line: int = 0,
) -> NDArray[np.int64]:
    """Start-to-start intervals of lines followed by a line of the same frame.

    The interval from start i to start i + 1 is dropped when a frame marker
    lies in (start i, start i + 1] (a marker coinciding with a line start
    precedes that line, as in ImageReconstructor), or, with
    ``lines_per_frame``, when line ``first_line + i + 1`` starts a frame by
    line count. Dropped intervals hold the flyback and any pause between
    frames, e.g. of a time-lapse, which are not line periods.
    """
    starts = _as_nsyncs("start_nsyncs", start_nsyncs)
    frames = _as_nsyncs("frame_nsyncs", frame_nsyncs)
    intervals = np.diff(starts)
    keep = np.ones(len(intervals), dtype=bool)
    lines_before = np.searchsorted(starts, frames, side="left")
    lines_before = lines_before[
        (lines_before >= 1) & (lines_before <= len(intervals))
    ]
    keep[lines_before - 1] = False
    if lines_per_frame:
        next_line = first_line + np.arange(1, len(starts), dtype=np.int64)
        keep &= next_line % lines_per_frame != 0
    return intervals[keep]


def estimate_line_duration(
    frame_nsyncs: NDArray[np.uint64],
    start_nsyncs: NDArray[np.uint64],
    stop_nsyncs: NDArray[np.uint64],
    *,
    lines_per_frame: int | None = None,
    first_line: int = 0,
) -> LineDurationEstimate:
    """Estimate the line duration from all lines of the given markers.

    With line stop markers, the median start-to-stop time; a pause between
    frames does not affect it. Without, a guess: GUESSED_LINE_PHASE times
    the median line period, from within-frame intervals only (see
    :func:`within_frame_intervals`).
    """
    periods = within_frame_intervals(
        start_nsyncs, frame_nsyncs, lines_per_frame, first_line
    )
    median_period = float(np.median(periods)) if len(periods) else None
    timing = analyze_stop_marker_timing(
        np.empty(0, dtype=np.int64), start_nsyncs, stop_nsyncs
    )
    if timing.pair_count:
        return LineDurationEstimate(
            int(timing.median_duration), "stop_markers", median_period
        )
    if median_period is not None:
        return LineDurationEstimate(
            int(median_period * GUESSED_LINE_PHASE), "line_period", median_period
        )
    return LineDurationEstimate(None, None, None)


def guessed_line_duration_issue(estimate: LineDurationEstimate) -> str | None:
    """A message if the line duration of ``estimate`` is a guess from the
    line period (no line stop markers); None otherwise."""
    if estimate.source != "line_period":
        return None
    return (
        f"No line stop markers: line_duration guessed as "
        f"{GUESSED_LINE_PHASE:.0%} of the line period, {estimate.duration} "
        f"of {estimate.median_period:.0f} (nsync units). Set "
        f"ScanConfig.line_duration if the scanner's value is known."
    )


def line_duration_issue(
    line_duration: int, median_period: float | None
) -> str | None:
    """A message if ``line_duration`` exceeds the median line period, so that
    consecutive line windows overlap; None otherwise."""
    if median_period is None or line_duration <= median_period:
        return None
    return (
        f"line_duration={line_duration} exceeds the line period "
        f"{median_period:.0f} (median start-to-start interval within a "
        f"frame, nsync units); consecutive line windows overlap."
    )
