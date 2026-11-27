"""First-frame pre-scan: check a ScanConfig against the data before
reconstruction, and estimate the line duration.

Reconstruction processes one chunk at a time and needs the line duration
from the start. The pre-scan gets it from the first frame by reading the
data once more, keeping only the markers (photons are decoded and dropped),
so its memory use does not depend on the photon count. It stops shortly
after the first frame, or after the first frames holding at least
PRESCAN_MIN_LINES lines when frames are small.

The same pass checks the frame geometry: with frame markers, a marker must
sit at every multiple of lines_per_frame (= lines x sum(line_accumulations))
line starts, whether the hardware marks the frame start or its end. Problems
are reported, not diagnosed; see :mod:`tttrkit.ptuio.diagnostics`.

Typical use::

    scan = prescan_first_frame(reader, config)  # fills config.line_duration
    rec = ImageReconstructor(config)
    for chunk in reader.iter_chunks():
        rec.update(corrector.correct(chunk))

prescan_events does the same for event chunks already decoded and
overflow-corrected, e.g. synthetic data.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import warnings

import numpy as np

from .config import ScanConfig, ScanConfigWarning
from .decoder import T3OverflowCorrector, resolve_markers
from .marker_timing import (
    estimate_line_duration,
    frame_boundary_mismatch,
    guessed_line_duration_issue,
    line_duration_issue,
)
from .reader import TTTRReader

# Small frames: the line duration is estimated from the first frames that
# hold at least this many lines, so that it rests on enough lines.
PRESCAN_MIN_LINES = 128


@dataclass
class FirstFrameScan:
    """Result of :func:`prescan_first_frame`.

    line_duration_source:
        "config" (given, checked against the line period), "stop_markers"
        or "line_period" (estimated), None if it could not be estimated.
    measured_lines_per_frame:
        Most common spacing of the frame markers, in line starts; None
        without two frame markers to measure (or with frame_sync="count").
    issues:
        One message per problem found, empty if the data agrees with the
        config; each was also issued as a ScanConfigWarning unless
        warn=False. A line duration guessed from the line period (no line
        stop markers) is one. Without frame markers the frames are not
        checked.
    """

    line_duration: int | None = None  # nsync units
    line_duration_source: str | None = None
    median_line_period: float | None = None  # within-frame lines, nsync units
    measured_lines_per_frame: int | None = None
    issues: list[str] = field(default_factory=list)


def prescan_first_frame(
    reader: TTTRReader,
    config: ScanConfig,
    *,
    chunk_size: int = 1_000_000,
    warn: bool = True,
) -> FirstFrameScan:
    """Check ``config`` against the first frame; fill in its line duration.

    Args:
        reader: Read from the start of the data with a fresh overflow
            corrector; its read position is left untouched.
        config: The ScanConfig to check. If ``config.line_duration`` is
            None, it is set to the estimate (if one could be made);
            otherwise it is checked: it must not exceed the line period.
            An estimate without line stop markers is a guess and is
            reported as an issue.
        chunk_size: Records per read.
        warn: Issue every problem found as a ScanConfigWarning.

    Returns:
        A FirstFrameScan. If the line duration could not be estimated,
        config.line_duration stays None and ImageReconstructor refuses it.
    """
    corrector = T3OverflowCorrector()
    raw = reader.iter_chunks(chunk_size, start=reader.offset)
    chunks = (corrector.correct(c) for c in raw)
    return _report(_scan(chunks, config), warn)


def prescan_events(
    chunks: Iterable[np.ndarray],
    config: ScanConfig,
    *,
    warn: bool = True,
) -> FirstFrameScan:
    """As :func:`prescan_first_frame`, for overflow-corrected event chunks
    (``event_dtype``) starting at the start of the data; they are consumed
    only as far as needed."""
    return _report(_scan(chunks, config), warn)


def _report(scan: FirstFrameScan, warn: bool) -> FirstFrameScan:
    if warn:
        for issue in scan.issues:
            # Attributed to the caller of the public function
            warnings.warn(issue, ScanConfigWarning, stacklevel=3)
    return scan


def _scan(chunks: Iterable[np.ndarray], config: ScanConfig) -> FirstFrameScan:
    lpf = config.lines_per_frame
    # Lines for the line duration: whole frames, at least PRESCAN_MIN_LINES
    estimate_lines = lpf * max(1, -(-PRESCAN_MIN_LINES // lpf))
    # Read on a little, to find a frame marker that comes a few lines late
    needed = estimate_lines + config.frame_sync_tolerance_lines + 1

    frames, starts, stops = [], [], []
    n_starts = 0
    for events in chunks:
        f, s, p = resolve_markers(
            events,
            config.frame_start_marker_channel,
            config.line_start_marker_channel,
            config.line_stop_marker_channel,
        )
        frames.append(f["nsync"])
        starts.append(s["nsync"])
        stops.append(p["nsync"])
        n_starts += len(s)
        if n_starts >= needed:
            break

    def joined(arrays):
        if not arrays:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(arrays).astype(np.int64)

    starts = joined(starts)[:needed]
    frames = joined(frames) if config.frame_sync != "count" else joined([])
    stops = joined(stops)
    end_of_data = len(starts) < needed
    if not end_of_data:
        # Markers after the last line start read cannot be placed yet
        frames = frames[frames <= starts[-1]]

    scan = FirstFrameScan()
    frame_lines = np.searchsorted(starts, frames, side="left")
    _check_frames(scan, config, frame_lines, len(starts), end_of_data)
    _line_duration(scan, config, frames, starts, stops, estimate_lines)
    return scan


def _check_frames(scan, config, frame_lines, n, end_of_data):
    """Compare the frame markers (line starts preceding each) with
    lines_per_frame."""
    lpf = config.lines_per_frame
    tolerance = config.frame_sync_tolerance_lines
    geometry = (
        f"lines_per_frame = lines x sum(line_accumulations) = "
        f"{config.lines} x {config.total_accumulations} = {lpf}"
    )
    positions = np.unique(frame_lines)
    # Line 0 starts frame 0, as in the frame sync. Spacings within the
    # tolerance (a glitch next to a marker) are no frame lengths.
    spacings = np.diff(np.union1d(0, positions))
    lengths = spacings[spacings > tolerance]
    if len(lengths):
        values, counts = np.unique(lengths, return_counts=True)
        scan.measured_lines_per_frame = int(values[np.argmax(counts)])

    if end_of_data and n < lpf:
        scan.issues.append(
            f"The data holds {n} line start markers, fewer than one frame "
            f"({geometry})."
        )
        return
    if len(positions) == 0:
        return  # nothing to check the frames against

    # Evenly spaced markers at a spacing other than lines_per_frame, e.g.
    # 0, 256, 512 for 512. A glitch breaks the even spacing; a spacing within
    # the tolerance of lines_per_frame (e.g. 399 for 400: a dropped line
    # marker) is left to the frame sync.
    if (
        len(lengths) == len(spacings) > 0
        and np.all(spacings == spacings[0])
        and abs(spacings[0] - lpf) > tolerance
    ):
        scan.issues.append(
            f"Frame markers are {spacings[0]} line starts apart, but "
            f"{geometry}."
        )
        return

    # The end of the data needs no marker (a frame-start marker would
    # follow it); otherwise a marker up to `tolerance` lines late must be
    # within the scanned lines.
    last = n - 1 if end_of_data else n - 1 - tolerance
    missing = _missing_boundary(positions, lpf, tolerance, last)
    if missing is not None:
        measured = scan.measured_lines_per_frame
        scan.issues.append(
            f"No frame marker within {tolerance} line(s) of line start "
            f"{missing} ({geometry})."
            + (f" Frame markers are {measured} line starts apart." if measured else "")
        )


def _missing_boundary(positions, lpf, tolerance, last):
    """First frame boundary up to line start ``last`` without a frame marker.

    Follows the frame sync (frame_sync="resync"): line 0 starts frame 0, and
    a marker at most ``tolerance`` lines off a boundary marks it and realigns
    the following boundaries; markers further off are glitches.
    """
    anchor, offset = 0, 0
    for p in positions.tolist():
        _, mismatch = frame_boundary_mismatch(p - offset, lpf)
        if abs(mismatch) > tolerance:
            continue
        if p > anchor + lpf + tolerance:
            return anchor + lpf
        anchor, offset = p, offset + mismatch
    return anchor + lpf if anchor + lpf <= last else None


def _line_duration(scan, config, frames, starts, stops, estimate_lines):
    # The first frame(s), plus the next line start to close the last line
    starts = starts[: estimate_lines + 1]
    if len(starts) > estimate_lines:
        frames = frames[frames <= starts[-1]]
        stops = stops[stops <= starts[-1]]
    estimate = estimate_line_duration(
        frames, starts, stops, lines_per_frame=config.lines_per_frame
    )
    scan.median_line_period = estimate.median_period

    if config.line_duration is not None:
        scan.line_duration = int(config.line_duration)
        scan.line_duration_source = "config"
        issue = line_duration_issue(scan.line_duration, estimate.median_period)
        if issue:
            scan.issues.append(issue)
    elif estimate.duration is None:
        scan.issues.append(
            f"The line duration could not be estimated from {len(starts)} "
            f"line start marker(s) (no line stop markers and no two "
            f"consecutive lines in a frame); set ScanConfig.line_duration."
        )
    else:
        scan.line_duration = config.line_duration = estimate.duration
        scan.line_duration_source = estimate.source
        issue = guessed_line_duration_issue(estimate)
        if issue:
            scan.issues.append(issue)
