import copy
from typing import Dict

import numpy as np
import xarray as xr
from scipy.signal import correlate
from .decoder import T3OverflowCorrector, resolve_markers
from .reader import TTTRReader
from .reconstructor import ScanConfig, SegmentReconstructor
from scipy.optimize import curve_fit

# --- Reconstruction helpers ---


def _gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def _fit_gaussian_peak(
    shifts: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray] | None:
    try:
        p0 = [
            scores.max() - scores.min(),
            shifts[np.argmax(scores)],
            0.1 * (shifts.max() - shifts.min()),
            scores.min(),
        ]
        popt, _ = curve_fit(_gaussian, shifts, scores, p0=p0)
        fit = _gaussian(shifts, *popt)
        return float(popt[1]), fit  # mu = estimated phase shift

    except RuntimeError:
        return None


def estimate_tcspc_bins(header_tags: dict, buffer: int = 10) -> int:
    rep_rate = header_tags.get("TTResult_SyncRate", 40e6)  # Hz
    resolution = header_tags.get("MeasDesc_Resolution", 5e-12)  # s
    bins = int(np.ceil(1 / resolution / rep_rate)) + buffer
    return bins


def _skipped_line_parity(corrected_chunk: np.ndarray, config: ScanConfig, parity: int) -> int:
    """Update the running forward/backward line parity (0=forward, 1=backward)
    across a corrected chunk that is being skipped (not reconstructed).

    ``parity`` tracks the direction of the *next, not-yet-seen* line: if an
    even number of lines have occurred so far (0-indexed lines 0, 1, ..., an
    even count), the next line is odd (backward), and vice versa - hence the
    final ``1 -`` inversion of the plain cumulative count.

    Scanning direction continues to alternate plainly across frame boundaries
    (a frame marker does not reset it back to forward), so this is a pure
    cumulative count of line-start markers, mod 2.

    Counting is based only on line-start markers (never stop markers), so a
    line that starts in this chunk but only stops in the next one is still
    counted exactly once here, and the stray stop marker at the start of the
    next chunk is correctly ignored there (each start marker begins exactly
    one line, regardless of where its matching stop falls).
    """
    _, start_markers, _ = resolve_markers(
        corrected_chunk,
        config.frame_start_marker_channel,
        config.line_start_marker_channel,
        config.line_stop_marker_channel,
    )
    return 1 - ((parity + len(start_markers)) % 2)


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


def estimate_bidirectional_prealign(
    reader: TTTRReader,
    cfg: ScanConfig,
    wrap: int = 1024,
    chunk_length: int = 500_000,
    skip_chunks: int = 0,
    verbose = True,
):
    probe_config = copy.deepcopy(cfg)
    probe_config.bidirectional_phase_shift = 0.0

    corrected_chunk, parity = _read_probe_chunk(
        reader, cfg, wrap, chunk_length, skip_chunks, verbose
    )

    _, start_markers, stop_markers = resolve_markers(
    corrected_chunk,
    4,
    1,
    2,
    )    

    start_marker_nsync = start_markers['nsync']
    stop_marker_nsync = stop_markers['nsync']
    stop_marker_nsync = stop_marker_nsync[stop_marker_nsync > start_marker_nsync[0]]

    n_pairs = min(len(start_marker_nsync),len(stop_marker_nsync))
    start_marker_nsync = start_marker_nsync[:n_pairs]
    stop_marker_nsync = stop_marker_nsync[:n_pairs]

    periods = np.diff(start_marker_nsync)
    durations = stop_marker_nsync - start_marker_nsync
    pauses = start_marker_nsync[1:] - stop_marker_nsync[:-1]
    print(f"{np.mean(durations)} +/- {np.std(durations)}")
    print(f"{np.mean(pauses)} +/- {np.std(pauses)}")
    print(f"Duty: {np.mean(durations[:-1] / periods)}")


    pause_phase = np.mean(pauses) / np.mean(durations) 
    pause_phase += cfg.line_start_marker_delay 
    pause_phase -= cfg.line_stop_marker_delay
    
    margin = 0.9*pause_phase /2
    probe_config.line_start_marker_delay += -margin
    probe_config.line_stop_marker_delay += margin

    probe_chunk, parity = _read_probe_chunk(
        reader=reader,
        config=probe_config,
        wrap=wrap,
        chunk_length=chunk_length,
        skip_chunks=skip_chunks,
        verbose=verbose
    )

    seg_recon = SegmentReconstructor(probe_config)
    ds = seg_recon.reconstruct(probe_chunk)

    n_lines = ds.sizes["line"]
    # Forward/backward alignment is determined purely by the parity carried
    # over from the skipped region: scanning direction alternates plainly and
    # is not reset by a frame marker, so a break inside the probe chunk itself
    # doesn't change which local line is really forward.
    start_line = parity
    if len(ds.frame_break_line) > 0 and verbose:
        print(f"Warning: {len(ds.frame_break_line.values)} frame break(s) detected in the chunk.")

    photon_count = ds.photon_count.isel(line=slice(start_line, n_lines)).values
    forward = photon_count[0::2].sum(axis=0).astype(np.float64)
    backward = photon_count[1::2].sum(axis=0).astype(np.float64)

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
    phase_shift = pixel_shift * 512 / (np.mean(durations) + 0.9 * np.mean(pauses)) / 2
    phase_shift /= (1-cfg.line_start_marker_delay)
    phase_shift /=(1+cfg.line_start_marker_delay)

    backward_aligned = np.roll(backward,pixel_shift)

    if verbose:
        print(f"Coarse forward/backward pixel offset: {pixel_shift}")
        print(f"Coarse bidirectional phase shift: {phase_shift:.5f}")

    return xr.Dataset(
        {
            "forward": (("pixel",), forward),
            "backward": (("pixel",), backward),
            "backward_aligned":(("pixel",), backward_aligned),
            "pixel_shift": ((), pixel_shift),
            "phase_shift": ((), phase_shift),
        },
        coords={"pixel": np.arange(probe_config.pixels)},
    )



def estimate_bidirectional_shift(
    reader: TTTRReader,
    config: ScanConfig,
    wrap: int = 1024,
    max_shift: float = 0.01,
    steps: int = 11,
    chunk_length: int = 500_000,
    skip_chunks: int = 0,
    verbose: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Estimate the optimal phase shift (as fraction of line duration) for backward lines
    in bidirectional scanning.

    Args:
        reader: TTTRReader instance
        config: A ScanConfig instance.
        max_shift: Maximum shift to try (±max_shift).
        steps: Number of shift steps to test.
        chunk_length: Number of events to read. Try increasing it when reconstruction fails, perhaps the reconstruction is feature-less
        skip_chunks: Number of leading chunks of chunk_length events to skip first, e.g. to
            probe the same region used by :func:`estimate_bidirectional_prealign`.
        verbose: Whether to print progress.

    Returns:
        Tuple of best phase shift (float) in units of line duration (e.g., -0.015) and numpy array of shifts, correlation scores, and fit for inspection.
    """

    if not config.bidirectional:
        raise ValueError(
            "ScanConfig must have bidirectional=True to estimate phase shift."
        )

    if verbose:
        print("Estimating bidirectional phase shift...")

    shifts = np.linspace(
        config.bidirectional_phase_shift - max_shift,
        config.bidirectional_phase_shift + max_shift,
        steps,
    )
    scores = np.zeros_like(shifts)

    # Read the probe chunk once and reuse it for every shift, so all shifts are
    # evaluated on the same data; parity tells us whether local line 0 is really
    # forward or backward (see estimate_bidirectional_prealign for the same logic).
    corrected_chunk, parity = _read_probe_chunk(
        reader, config, wrap, chunk_length, skip_chunks, verbose
    )

    for i, shift in enumerate(shifts):

        # Clone config and apply shift
        test_config = copy.deepcopy(config)
        test_config.bidirectional_phase_shift = shift
        seg_recon = SegmentReconstructor(test_config)
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
                print(f"Shift {shift:.4f} → no usable line pairs")
            continue

        # Subtract mean along each line (axis=1)
        fwd_vals = fwd_vals - fwd_vals.mean(axis=1, keepdims=True)
        bwd_vals = bwd_vals - bwd_vals.mean(axis=1, keepdims=True)

        # Compute dot products (correlation at lag zero)
        score = np.sum(fwd_vals * bwd_vals)

        scores[i] = score

        if verbose:
            print(f"Shift {shift:.4f} → score {score:.2f}")

    # best_shift = shifts[np.argmax(scores)]

    fit_result = _fit_gaussian_peak(shifts, scores)
    if fit_result is None:
        best_shift = float(shifts[np.argmax(scores)])
        fit = np.full_like(scores, np.nan)
    else:
        best_shift, fit = fit_result

    if verbose:
        print(f"Best estimated shift: {best_shift:.5f}")

    return best_shift, np.stack((shifts, scores, fit))


# --- Marker Helpers ---


def marker_events(events: np.ndarray) -> np.ndarray:
    """Return only events where channel == 63 and special != 15 (non-overflow markers)."""
    return events[(events["channel"] < 63) & (events["special"] != 0)]


def get_marker_distribution(events: np.ndarray) -> Dict[int, int]:
    """Returns a count of each special marker code."""
    mask = (events["channel"] < 63) & (events["special"] != 0)
    markers = events["channel"][mask]
    unique, counts = np.unique(markers, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

# Backward-compatible imports for existing notebooks and user code.
# from ..analysis.decay import shift_decay
# from ..analysis.phasor import average_phasor, get_phasor_from_decay, smooth_phasor
# from ..analysis.visualization import create_FLIM_image, draw_unitary_circle


