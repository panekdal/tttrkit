"""Analysis helpers for frame and line timing markers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StopMarkerTiming:
    """Timing measurements derived from line-start and line-stop markers."""

    pair_count: int
    phases: NDArray[np.float64]
    durations: NDArray[np.int64]
    intervals: NDArray[np.int64]
    paired_intervals: NDArray[np.int64]
    median_phase: float | None
    median_duration: float | None
    median_interval: float | None
    window: tuple[int | None, int | None]


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

    if len(frame) >= 2:
        lower, upper = int(frame[0]), int(frame[1])
    elif len(frame) == 1:
        lower, upper = int(frame[0]), None
    else:
        lower, upper = None, None

    if lower is not None:
        starts = starts[starts >= lower]
        stops = stops[stops >= lower]
    if upper is not None:
        starts = starts[starts < upper]
        stops = stops[stops < upper]

    intervals = np.diff(starts)
    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=np.float64)

    if len(intervals) == 0:
        return StopMarkerTiming(
            pair_count=0,
            phases=empty_float,
            durations=empty_int,
            intervals=intervals,
            paired_intervals=empty_int,
            median_phase=None,
            median_duration=None,
            median_interval=None,
            window=(lower, upper),
        )

    # Each line can use only a stop after its start and before its next start.
    stop_indices = np.searchsorted(stops, starts[:-1], side="right")
    paired = stop_indices < len(stops)
    paired[paired] &= stops[stop_indices[paired]] < starts[1:][paired]

    paired_starts = starts[:-1][paired]
    paired_stops = stops[stop_indices[paired]]
    durations = paired_stops - paired_starts
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
        intervals=intervals,
        paired_intervals=paired_intervals,
        median_phase=float(np.median(phases)) if len(phases) else None,
        median_duration=float(np.median(durations)) if len(durations) else None,
        median_interval=float(np.median(intervals)),
        window=(lower, upper),
    )

def compute_line_duration(
    frame_nsyncs,
    start_nsyncs,
    stop_nsyncs,
    default_phase: float = 0.80,
):
    timing = analyze_stop_marker_timing(
        frame_nsyncs, start_nsyncs, stop_nsyncs
    )

    if timing.pair_count:
        return timing.median_phase, int(timing.median_duration)

    if timing.median_interval is not None:
        return default_phase, int(timing.median_interval * default_phase)

    return None, None