"""Acquisition parameters estimated from the data: TCSPC bins and the
bidirectional forward/backward alignment."""

import copy

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import correlate

from .config import ScanConfig
from .decoder import T3OverflowCorrector, resolve_markers
from .marker_timing import analyze_stop_marker_timing
from .prescan import prescan_first_frame
from .reader import TTTRReader
from .reconstructor import SegmentReconstructor


def estimate_tcspc_bins(header_tags: dict, buffer: int = 10) -> int:
    rep_rate = header_tags.get("TTResult_SyncRate", 40e6)  # Hz
    resolution = header_tags.get("MeasDesc_Resolution", 5e-12)  # s
    bins = int(np.ceil(1 / resolution / rep_rate)) + buffer
    return bins


def _gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def _fit_gaussian_peak(
    shifts: np.ndarray,
    scores: np.ndarray,
    number_of_plot_points: int = 100,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    try:
        p0 = [
            scores.max() - scores.min(),
            shifts[np.argmax(scores)],
            0.1 * (shifts.max() - shifts.min()),
            scores.min(),
        ]
        popt, _ = curve_fit(_gaussian, shifts, scores, p0=p0)
        fit_shifts = np.linspace(shifts[0], shifts[-1], number_of_plot_points)

        fit = _gaussian(fit_shifts, *popt)
        return float(popt[1]), fit_shifts, fit  # mu = estimated phase shift

    except RuntimeError:
        return None


def _skipped_line_parity(corrected_chunk: np.ndarray, config: ScanConfig, parity: int) -> int:
    """Update the running forward/backward line parity (0=forward, 1=backward)
    across a corrected chunk that is being skipped (not reconstructed).

    ``parity`` is the direction of the *next, not-yet-seen* line: the number
    of lines so far, mod 2. Scanning direction continues to alternate plainly
    across frame boundaries (a frame marker does not reset it back to
    forward), so this is a pure cumulative count of line-start markers.

    Counting is based only on line-start markers (never stop markers), so a
    line that starts in this chunk but only stops in the next one is still
    counted exactly once here, and the stray stop marker at the start of the
    next chunk is correctly ignored there.
    """
    _, start_markers, _ = resolve_markers(
        corrected_chunk,
        config.frame_start_marker_channel,
        config.line_start_marker_channel,
        config.line_stop_marker_channel,
    )
    return (parity + len(start_markers)) % 2


def _read_probe_chunk(
    reader: TTTRReader,
    config: ScanConfig,
    wrap: int,
    chunk_length: int,
    skip_chunks: int,
    verbose: bool,
) -> tuple[np.ndarray, int]:
    """Reset the reader, skip ``skip_chunks`` chunks (tracking forward/backward
    line parity through them, since overflow correction is stateful and must
    still process them), then read+correct one more chunk to probe/test.

    Returns the corrected probe chunk and the parity (0=forward, 1=backward)
    of its first line, needed to know whether local line index 0 is really
    forward or backward.
    """
    reader.reset()
    corrector = T3OverflowCorrector(wraparound=wrap)

    parity = 0
    for i in range(skip_chunks):
        skipped_chunk = reader.read(count=chunk_length)
        corrected_skipped = corrector.correct(skipped_chunk)
        parity = _skipped_line_parity(corrected_skipped, config, parity)
        if verbose:
            print(f"Skipped chunk {i + 1}/{skip_chunks}")

    chunk = reader.read(count=chunk_length)
    corrected_chunk = corrector.correct(chunk)
    return corrected_chunk, parity


def _with_line_duration(
    reader: TTTRReader, config: ScanConfig, wrap: int
) -> ScanConfig:
    """A copy of config with the line duration of the first frame filled in
    (see prescan_first_frame) if it is not set, so that segment
    reconstruction uses the same line duration as image reconstruction."""
    config = copy.deepcopy(config)
    if config.line_duration is None:
        prescan_first_frame(reader, config, wrap=wrap)
        if config.line_duration is None:
            raise ValueError(
                "The line duration could not be estimated from the first "
                "frame; set ScanConfig.line_duration."
            )
    return config


def estimate_bidirectional_prealign(
    reader: TTTRReader,
    cfg: ScanConfig,
    laser_sync_rate: float,
    wrap: int = 1024,
    chunk_length: int = 500_000,
    skip_chunks: int = 0,
    verbose = True,
) -> xr.Dataset:
    probe_config = _with_line_duration(reader, cfg, wrap)

    corrected_chunk, parity = _read_probe_chunk(
        reader, cfg, wrap, chunk_length, skip_chunks, verbose
    )

    frame_markers, start_markers, stop_markers = resolve_markers(
        corrected_chunk,
        cfg.frame_start_marker_channel,
        cfg.line_start_marker_channel,
        cfg.line_stop_marker_channel,
    )

    marker_timing = analyze_stop_marker_timing(
        frame_markers["nsync"],
        start_markers["nsync"],
        stop_markers["nsync"],
    )
    if marker_timing.pair_count == 0:
        raise ValueError(
            "No line with both a start and a stop marker in the probe chunk; "
            "pre-alignment needs line stop markers. Check "
            "line_stop_marker_channel, or try a larger chunk_length."
        )
    # The line duration the segments are built with (first frame or given)
    duration_nsync = int(probe_config.line_duration)
    # Durations exist only for lines with a stop marker; pair them with
    # their own intervals (lines without a stop are left out of both)
    pauses_nsync = marker_timing.paired_intervals - marker_timing.durations
    if verbose:
        print(f"Durations [nsync]: {marker_timing.median_duration} +/- {np.std(marker_timing.durations)}")
        print(f"Pauses [nsync]: {np.median(pauses_nsync)} +/- {np.std(pauses_nsync)}")
        print(f"Duty: {marker_timing.median_phase}")

    # add margins to the reconstructed lines duration
    # ignore the input delays
    margin_nsync = int(np.median(pauses_nsync))

    probe_config.line_start_marker_delay = -int(margin_nsync /2 )
    probe_config.line_stop_marker_delay = int(margin_nsync /2)

    window_nsync = margin_nsync + duration_nsync

    # increase proportionally the number of pixels
    probe_config.pixels = int(np.ceil(cfg.pixels * (window_nsync) / duration_nsync))
    pixel = np.arange(probe_config.pixels)

    single_pixel_duration_nsync = int((window_nsync) / probe_config.pixels)

    time_axis = pixel * single_pixel_duration_nsync / laser_sync_rate

    # shift the time axis so it coincides with the start marker(s)
    time_axis -= margin_nsync / laser_sync_rate / 2

    probe_chunk, parity = _read_probe_chunk(
        reader,
        probe_config,
        wrap,
        chunk_length,
        skip_chunks,
        verbose
    )

    seg_recon = SegmentReconstructor(probe_config,laser_sync_rate)
    ds = seg_recon.reconstruct(probe_chunk)

    n_lines = ds.sizes["line"]
    # Forward/backward alignment is determined purely by the parity carried
    # over from the skipped region: scanning direction alternates plainly and
    # is not reset by a frame marker, so a break inside the probe chunk itself
    # doesn't change which local line is really forward.
    if len(ds.frame_break_line) > 0 and verbose:
        print(f"Warning: {len(ds.frame_break_line.values)} frame break(s) detected in the chunk.")

    photon_count = ds.photon_count.values[:n_lines]

    n = len(photon_count) // 2
    odd_line_sum = photon_count[0:2*n:2].sum(axis=0)
    even_line_sum = photon_count[1:2*n:2].sum(axis=0)

    if parity == 0:
        forward = odd_line_sum
        backward = even_line_sum
    else:
        forward = even_line_sum[::-1]
        backward = odd_line_sum[::-1]


    if len(forward) == 0 or len(backward) == 0:
        raise ValueError(
            "Not enough complete forward/backward line pairs in the probe chunk. "
            "Try increasing chunk_length."
        )

    cross_corr = correlate(
        forward - forward.mean(), backward - backward.mean(), mode="full", method="fft"
    )
    lags = np.arange(-len(forward) + 1, len(forward))
    pixel_shift = int(lags[np.argmax(cross_corr)])
    time_shift = pixel_shift * single_pixel_duration_nsync / laser_sync_rate

    durations_s = marker_timing.durations / laser_sync_rate
    periods_s = marker_timing.paired_intervals / laser_sync_rate

    backward_aligned = np.roll(backward,pixel_shift)

    if verbose:
        print(f"Coarse forward/backward pixel offset: {pixel_shift}")
        print(f"Coarse shift [μs]: {1e6 * time_shift:.3f}")

    return xr.Dataset(
        {
            "forward": (("pixel",), forward),
            "backward": (("pixel",), backward),
            "backward_aligned":(("pixel",), backward_aligned),
            "durations_nsync": (("paired_interval"), marker_timing.durations),
            "periods_nsync": (("paired_interval"), marker_timing.paired_intervals),
            "durations_s": (("paired_interval"), durations_s),
            "periods_s": (("paired_interval"), periods_s),
            "pixel_shift": ((), pixel_shift),
            "time_shift": ((), time_shift),
        },
        coords={"pixel": pixel,
                "time_axis": time_axis,
                "paired_interval": np.arange(marker_timing.pair_count),
                },
    )

def estimate_bidirectional_shift(
    reader: TTTRReader,
    config: ScanConfig,
    laser_sync_rate: float,
    wrap: int = 1024,
    max_shift: int = 500, # in nsync
    steps: int = 11,
    chunk_length: int = 500_000,
    skip_chunks: int = 0,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Estimate the optimal phase shift (as fraction of line duration) for backward lines
    in bidirectional scanning.

    Args:
        reader: TTTRReader instance
        config: A ScanConfig instance.
        laser_sync_rate: Repetition rate of the sync signal in Hz
        wrap: .ptu wrap-around
        max_shift: Maximum shift to try (±max_shift).
        steps: Number of shift steps to test.
        chunk_length: Number of events to read. Try increasing it when reconstruction fails, perhaps the reconstruction is feature-less
        skip_chunks: Number of leading chunks of chunk_length events to skip first, e.g. to
            probe the same region used by :func:`estimate_bidirectional_prealign`.
        verbose: Whether to print progress.

    Returns:
        XArray dataset:
        best_shift: optimized shift value in s,
        scores: correlation amplitude for each shift,
        fit: values of the Gaussian fit,
    """

    if not config.bidirectional:
        raise ValueError(
            "ScanConfig must have bidirectional=True to estimate phase shift."
        )

    if verbose:
        print("Estimating bidirectional phase shift...")

    n_intervals = round((2 * max_shift) / (steps - 1))

    steps_adjusted = (2 * max_shift) // n_intervals + 1

    shifts = np.linspace(
        -max_shift,
        +max_shift,
        steps_adjusted,
        dtype=int,
    )

    scores = np.zeros_like(shifts)

    config = _with_line_duration(reader, config, wrap)

    # Read the probe chunk once and reuse it for every shift, so all shifts are
    # evaluated on the same data; parity tells us whether local line 0 is really
    # forward or backward (see estimate_bidirectional_prealign for the same logic).
    corrected_chunk, parity = _read_probe_chunk(
        reader, config, wrap, chunk_length, skip_chunks, verbose
    )

    for i, shift in enumerate(shifts):

        # Clone config and apply shift
        test_config = copy.deepcopy(config)
        test_config.line_start_marker_delay += shift
        test_config.line_stop_marker_delay += shift
        seg_recon = SegmentReconstructor(test_config,laser_sync_rate)
        ds = seg_recon.reconstruct(corrected_chunk)

        photon_count = ds.photon_count.values.astype(np.float32)
        if parity == 0:
            fwd_vals = photon_count[0::2]
            bwd_vals = photon_count[1::2]
        else:
            fwd_vals = photon_count[1::2]
            bwd_vals = photon_count[0::2]

        # Ensure same number of lines
        num_pairs = min(len(fwd_vals), len(bwd_vals))
        fwd_vals = fwd_vals[:num_pairs]
        bwd_vals = bwd_vals[:num_pairs]

        # Mask out zero rows
        mask = ~((fwd_vals == 0).all(axis=1) | (bwd_vals == 0).all(axis=1))
        fwd_vals = fwd_vals[mask]
        bwd_vals = bwd_vals[mask]

        if len(fwd_vals) == 0:
            scores[i] = 0.0
            if verbose:
                print(f"Shift {shift / laser_sync_rate * 1e6:.2f} μs → no usable line pairs")
            continue

        # Subtract mean along each line (axis=1)
        fwd_vals = fwd_vals - fwd_vals.mean(axis=1, keepdims=True)
        bwd_vals = bwd_vals - bwd_vals.mean(axis=1, keepdims=True)

        # Compute dot products (correlation at lag zero)
        score = np.sum(fwd_vals * bwd_vals)

        scores[i] = score

        if verbose:
            print(f"Shift {shift / laser_sync_rate * 1e6:.2f} μs → score {score:.2f}")

    fit_result = _fit_gaussian_peak(shifts, scores)
    if fit_result is None:
        best_shift = float(shifts[np.argmax(scores)])
        fit_shifts = np.full_like(scores, np.nan, dtype=float)
        fit = np.full_like(scores, np.nan, dtype=float)
    else:
        best_shift, fit_shifts, fit = fit_result
        best_shift = int(np.round(best_shift).item())

    if verbose:
        print(f"Best estimated shift: {best_shift / laser_sync_rate * 1e6:.3f} μs")

    return xr.Dataset(
        {
            "best_shift": ((),best_shift),
            "scores": (("test_shift",),scores),
            "fit": (("fit_shift",),fit),
        },
        coords={
            "test_shift": shifts,
            "fit_shift": fit_shifts,
        }
    )
