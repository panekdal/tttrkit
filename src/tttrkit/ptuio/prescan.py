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
line starts, whether the hardware marks the frame start or its end.

Typical use::

    scan = prescan_first_frame(reader, config)  # fills config.line_duration
    rec = ImageReconstructor(config)
    for chunk in reader.iter_chunks():
        rec.update(corrector.correct(chunk))
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import warnings

import numpy as np

from .decoder import T3OverflowCorrector, resolve_markers
from .marker_timing import estimate_line_duration
from .reader import TTTRReader
from .reconstructor import ScanConfig, ScanConfigWarning, frame_boundary_mismatch

# Small frames: the line duration is estimated from the first frames that
# hold at least this many lines, so that it rests on enough lines.
PRESCAN_MIN_LINES = 128

VERDICTS = ("ok", "mismatch", "too_few_lines", "unverified")


@dataclass
class FirstFrameScan:
    """Result of :func:`prescan_first_frame`.

    verdict:
        "ok"             the frame markers (or, at the end of the data, the
                         line count) agree with lines_per_frame
        "mismatch"       they do not; see ``issues`` and ``suggestions``
        "too_few_lines"  the data ends before one frame is complete
        "unverified"     no frame markers to check against (or
                         frame_sync="count")
    line_duration_source:
        "config" (given, checked against the line period), "stop_markers"
        or "line_period" (estimated), None if it could not be estimated.
    issues:
        One message per problem found; each was also issued as a
        ScanConfigWarning unless warn=False.
    """

    lines_per_frame: int
    line_starts: int  # line start markers read
    end_of_data: bool  # the data ended before the scan was complete
    frame_marker_lines: list[int]  # line starts preceding each frame marker
    measured_lines_per_frame: int | None
    verdict: str
    line_duration: int | None  # nsync units
    line_duration_source: str | None
    median_line_period: float | None  # within-frame lines, nsync units
    issues: list[str] = field(default_factory=list)
    # (lines, total accumulations) pairs matching measured_lines_per_frame
    suggestions: list[tuple[int, int]] = field(default_factory=list)


def prescan_first_frame(
    source: TTTRReader | Iterable[np.ndarray],
    config: ScanConfig,
    *,
    wrap: int | None = None,
    chunk_size: int = 1_000_000,
    warn: bool = True,
) -> FirstFrameScan:
    """Check ``config`` against the first frame; fill in its line duration.

    Args:
        source: A TTTRReader, read from the start of the data with a fresh
            overflow corrector (its read position is restored afterwards),
            or an iterable of overflow-corrected event chunks
            (``event_dtype``), consumed only as far as needed.
        config: The ScanConfig to check. If ``config.line_duration`` is
            None, it is set to the estimate (if one could be made);
            otherwise it is checked: it must not exceed the line period.
        wrap: Overflow wraparound for a reader source (default: the header
            tag TTResultFormat_WrapAround, else 1024).
        chunk_size: Records per read for a reader source.
        warn: Issue every problem found as a ScanConfigWarning.

    Returns:
        A FirstFrameScan. If the line duration could not be estimated,
        config.line_duration stays None and ImageReconstructor refuses it.
    """
    if isinstance(source, TTTRReader):
        if wrap is None:
            wrap = source.header.tags.get("TTResultFormat_WrapAround", 1024)
        corrector = T3OverflowCorrector(wraparound=wrap)
        position = source._pos
        source.reset()
        raw = source.iter_chunks(chunk_size)
        try:
            scan = _scan((corrector.correct(c) for c in raw), config)
        finally:
            raw.close()
            source._pos = position
    else:
        scan = _scan(source, config)

    if warn:
        for issue in scan.issues:
            warnings.warn(issue, ScanConfigWarning, stacklevel=2)
    return scan


def _scan(chunks: Iterable[np.ndarray], config: ScanConfig) -> FirstFrameScan:
    lpf = config.lines_per_frame
    use_frame_markers = config.frame_sync != "count"
    # Lines for the line duration: whole frames, at least PRESCAN_MIN_LINES
    estimate_lines = lpf * max(1, -(-PRESCAN_MIN_LINES // lpf))
    # Read on a little, to find a frame marker that comes a few lines late
    tolerance = int(np.ceil(max(1.0, config.frame_sync_tolerance * lpf)))
    needed = estimate_lines + tolerance + 1

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
    frames = joined(frames) if use_frame_markers else np.empty(0, np.int64)
    stops = joined(stops)
    end_of_data = len(starts) < needed
    if not end_of_data:
        # Markers after the last line start read cannot be placed yet
        frames = frames[frames <= starts[-1]]
    frame_lines = np.searchsorted(starts, frames, side="left")

    scan = FirstFrameScan(
        lines_per_frame=lpf,
        line_starts=len(starts),
        end_of_data=end_of_data,
        frame_marker_lines=frame_lines.tolist(),
        measured_lines_per_frame=None,
        verdict="ok",
        line_duration=None,
        line_duration_source=None,
        median_line_period=None,
    )
    _check_frames(scan, config, frame_lines, tolerance)
    _line_duration(scan, config, frames, starts, stops, estimate_lines)
    return scan


def _measured_lines_per_frame(positions: np.ndarray, tolerance: int):
    """Most common spacing of frame markers, in line starts. Spacings within
    the tolerance (a glitch next to a marker) are not frame lengths."""
    spacings = np.diff(positions)
    spacings = spacings[spacings > tolerance]
    if len(spacings):
        values, counts = np.unique(spacings, return_counts=True)
        return int(values[np.argmax(counts)])
    if len(positions) == 1 and positions[0] > 0:
        return int(positions[0])
    return None


def _missing_boundary(positions, lpf, tolerance, n, end_of_data):
    """First frame boundary within the scanned lines without a frame marker.

    Follows the reconstruction (frame_sync="resync"): line 0 starts frame 0,
    and a marker at most `tolerance` lines off a boundary realigns the line
    count, so it marks a boundary; markers further off are glitches. After
    each such marker, the next one must follow lines_per_frame (within the
    tolerance) later, unless the scanned lines end before.
    """
    anchors, offset = [0], 0
    for p in positions.tolist():
        _, mismatch = frame_boundary_mismatch(p - offset, lpf)
        if abs(mismatch) <= tolerance:
            offset += mismatch
            anchors.append(p)
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b > a + lpf + tolerance:
            return a + lpf
    following = anchors[-1] + lpf
    # The end of the data needs no marker (a frame-start marker would
    # follow it); otherwise a marker up to `tolerance` lines late must be
    # within the scanned lines.
    last = n - 1 if end_of_data else n - 1 - tolerance
    return following if following <= last else None


def _suggestions(measured: int, config: ScanConfig) -> list[tuple[int, int]]:
    """(lines, total accumulations) with lines x accumulations = measured,
    the current total accumulations first."""
    current = config._total_accumulations
    totals = [current] + [a for a in range(1, 65) if a != current]
    return [
        (measured // a, a)
        for a in totals
        if measured % a == 0 and measured // a >= 8
    ]


def _check_frames(scan, config, frame_lines, tolerance):
    lpf = scan.lines_per_frame
    n = scan.line_starts
    geometry = (
        f"lines_per_frame = lines x sum(line_accumulations) = "
        f"{config.lines} x {config._total_accumulations} = {lpf}"
    )
    positions = np.unique(frame_lines)
    measured = _measured_lines_per_frame(positions, tolerance)
    scan.measured_lines_per_frame = measured

    if scan.end_of_data and n < 2:
        scan.verdict = "too_few_lines"
        scan.issues.append(
            f"The data holds {n} line start marker(s) on "
            f"line_start_marker_channel={config.line_start_marker_channel}; "
            f"check the marker channels."
        )
        return
    if scan.end_of_data and n < lpf:
        scan.verdict = "too_few_lines"
        scan.issues.append(
            f"The data holds {n} line start markers, fewer than one frame "
            f"({geometry})."
            + (
                f" Frame markers are {measured} lines apart."
                if measured
                else ""
            )
        )
    elif len(positions) == 0:
        if scan.end_of_data and n % lpf == 0:
            scan.verdict = "ok"  # whole frames by line count
        else:
            scan.verdict = "unverified"
        return
    else:
        missing = _missing_boundary(
            positions, lpf, tolerance, n, scan.end_of_data
        )
        # Markers at a different, regular spacing throughout the scanned
        # lines, e.g. 0, 256, 512 for lines_per_frame = 512. A glitch is
        # not evenly spaced with the other markers. A spacing within the
        # tolerance (e.g. 399 for 400: a dropped line marker, or one line
        # too many configured) is left to the frame sync, which realigns
        # and reports it frame by frame.
        regular = (
            len(positions) >= 2
            and measured is not None
            and abs(measured - lpf) > tolerance
            and np.all(np.diff(positions) == measured)
            and positions[-1] + measured > n - 1
        )
        if missing is None and not regular:
            # Off-boundary markers among on-boundary ones are glitches,
            # handled (and reported) by the reconstruction's frame sync
            scan.verdict = "ok"
            return
        scan.verdict = "mismatch"
        if measured == lpf:
            offset = int(positions[0] % lpf)
            scan.issues.append(
                f"Frame markers are {lpf} lines apart as configured, but "
                f"{offset} line(s) off the frame boundaries counted from the "
                f"first line start: the data does not begin with a frame. "
                f"Lines are assigned to frames by count ({geometry})."
            )
            return
        if measured is not None:
            scan.issues.append(
                f"Frame markers are {measured} line starts apart, but "
                f"{geometry}."
            )
        else:
            scan.issues.append(
                f"No frame marker after {missing} line starts "
                f"({geometry}); frames hold more lines than configured."
            )
    if measured and measured != lpf:
        scan.suggestions = _suggestions(measured, config)
        if scan.suggestions:
            scan.issues[-1] += " Matching (lines, accumulations): " + ", ".join(
                f"({l}, {a})" for l, a in scan.suggestions[:6]
            ) + "."


def _line_duration(scan, config, frames, starts, stops, estimate_lines):
    # The first frame(s), plus the next line start to close the last line
    starts = starts[: estimate_lines + 1]
    if len(starts) > estimate_lines:
        frames = frames[frames <= starts[-1]]
        stops = stops[stops <= starts[-1]]
    estimate = estimate_line_duration(
        frames, starts, stops, lines_per_frame=scan.lines_per_frame
    )
    scan.median_line_period = estimate.median_period

    if config.line_duration is not None:
        scan.line_duration = int(config.line_duration)
        scan.line_duration_source = "config"
        period = estimate.median_period
        if period is not None and config.line_duration > period:
            scan.issues.append(
                f"line_duration={config.line_duration} exceeds the line "
                f"period {period:.0f} (median start-to-start interval within "
                f"the first frame, nsync units); consecutive line windows "
                f"overlap."
            )
        return

    if estimate.duration is None:
        scan.issues.append(
            f"The line duration could not be estimated from "
            f"{len(starts)} line start marker(s) of the first frame "
            f"(no line stop markers and no two consecutive lines in a "
            f"frame); set ScanConfig.line_duration."
        )
        return
    scan.line_duration = estimate.duration
    scan.line_duration_source = estimate.source
    config.line_duration = estimate.duration
