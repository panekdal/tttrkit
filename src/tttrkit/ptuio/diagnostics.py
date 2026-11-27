"""Marker diagnostics, for data that does not reconstruct as expected."""

from collections import Counter

import numpy as np

from .config import (
    DEFAULT_FRAME_START_MARKER_CHANNEL,
    DEFAULT_LINE_START_MARKER_CHANNEL,
    DEFAULT_LINE_STOP_MARKER_CHANNEL,
)
from .decoder import T3OverflowCorrector, marker_events, resolve_markers
from .reader import TTTRReader


def count_marker_values(events: np.ndarray) -> dict[int, int]:
    """{marker value: count} of the marker events."""
    markers = marker_events(events)["channel"]
    unique, counts = np.unique(markers, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def analyze_markers(
        reader: TTTRReader,
        max_accumulations: int = 64,
        verbose = True,
        *,
        frame_start_marker_channel: int = DEFAULT_FRAME_START_MARKER_CHANNEL,
        line_start_marker_channel: int = DEFAULT_LINE_START_MARKER_CHANNEL,
        line_stop_marker_channel: int = DEFAULT_LINE_STOP_MARKER_CHANNEL,
) -> dict:
    """Count the markers of the whole file, per raw marker value and per
    role (with the given marker channels), and list the (lines,
    accumulations) pairs matching the line starts per frame marker."""
    start_marker_count: int = 0
    stop_marker_count: int = 0
    frame_marker_count: int = 0
    marker_distribution = Counter()

    corrector = T3OverflowCorrector()

    reader.reset()
    for chunk in reader.iter_chunks():
        corrected_chunk = corrector.correct(chunk)
        raw_markers = marker_events(corrected_chunk)
        marker_distribution.update(count_marker_values(raw_markers))
        frame_markers, start_markers, stop_markers = resolve_markers(
            corrected_chunk,
            frame_start_marker_channel,
            line_start_marker_channel,
            line_stop_marker_channel,
        )
        start_marker_count += len(start_markers)
        stop_marker_count += len(stop_markers)
        frame_marker_count += len(frame_markers)

    marker_distribution = dict(marker_distribution)
    frames = max(1, frame_marker_count) # suppose at least one frame, if it is image, even when there are no markers
    lines_per_frame = start_marker_count // frames

    suggestion_pairs = []
    for i in range(1, max_accumulations + 1):
        if lines_per_frame % i == 0:
            lines = lines_per_frame // i
            if 64 <= lines <= 4096:
                suggestion_pairs.append((lines, i))


    result = {
        "line_start_count": start_marker_count,
        "line_stop_count": stop_marker_count,
        "frame_count": frame_marker_count,
        "raw_marker_distribution": marker_distribution,
        "suggested_combinations": suggestion_pairs,
    }

    if verbose: print(_format_marker_suggestions(result))
    return result

def _format_marker_suggestions(analysis_results: dict) -> str:
    lines = []
    lines.append("\n=== ALL MARKER EVENTS ===")
    lines.append(f"{'  Channel':<12}{'Events detected':<12}")
    lines.append("-" * 34)

    for k, v in analysis_results["raw_marker_distribution"].items():
        lines.append(f"  {k:<12}{v:<12}")

    lines.append("\n=== RESOLVED MARKERS ===")
    lines.append(
        f"{'  Frame starts:':<15} {analysis_results['frame_count']:<2}"
    )
    lines.append(
        f"{'  Line starts:':<15} {analysis_results['line_start_count']:<2}"
    )
    lines.append(
        f"{'  Line stops:':<15} {analysis_results['line_stop_count']:<2}"
    )

    lines.append("\n=== POSSIBLE COMBINATIONS ===")
    lines.append(f"{'  Lines':<12}{'Accumulations':<12}")
    lines.append("-" * 34)

    for k, v in analysis_results["suggested_combinations"]:
        lines.append(f"  {k:<12}{v:<12}")

    return "\n".join(lines)
