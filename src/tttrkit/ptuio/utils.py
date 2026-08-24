import copy
from typing import Dict

import numpy as np
import xarray as xr
from .decoder import T3OverflowCorrector
from .reader import TTTRReader
from .reconstructor import ImageReconstructor, ScanConfig
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


def estimate_bidirectional_shift(
    reader: TTTRReader,
    config: ScanConfig,
    wrap: int = 1024,
    max_shift: float = 0.01,
    steps: int = 11,
    chunk_length: int = 500_000,
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

    base_config = copy.deepcopy(config)
    base_config.frames = 1
    base_config.line_accumulations = (1,)
    base_config.lines = config.lines * config.line_accumulations[0]
    base_config._total_accumulations = 1

    line_bin = config.line_accumulations[0] * 2

    shifts = np.linspace(
        config.bidirectional_phase_shift - max_shift,
        config.bidirectional_phase_shift + max_shift,
        steps,
    )
    scores = np.zeros_like(shifts)
    corrector = T3OverflowCorrector(wraparound=wrap)

    for i, shift in enumerate(shifts):

        # Clone config and apply shift
        test_config = copy.deepcopy(base_config)
        test_config.bidirectional_phase_shift = shift
        recon = ImageReconstructor(
            config=test_config, outputs=["photon_count"]
        )
        chunk = reader.read(count=chunk_length)
        corrected_chunk = corrector.correct(chunk)
        recon.update(corrected_chunk)
        pc = xr.DataArray(
            data=recon.photon_count.astype(np.float32),
            coords={
                "frame": np.arange(test_config.frames),
                "line": np.arange(test_config.lines),
                "pixel": np.arange(test_config.pixels),
                "channel": np.arange(test_config.max_detector),
            },
        )

        # pc = xr.DataArray(recon.photon_count.astype(np.float32))
        # pc = pc.rename({"dim_0" : "frame",
        #     "dim_1" : "line",
        #     "dim_2" : "pixel",
        #     "dim_3" : "channel"})
        pc = pc.sum(dim="channel")
        pc = pc.isel(frame=0)
        forward = pc[::2, :]
        backward = pc[1::2, :]
        forward = forward.coarsen(line=line_bin).sum()
        backward = backward.coarsen(line=line_bin).sum()

        # Ensure same number of lines
        num_pairs = min(forward.sizes["line"], backward.sizes["line"])
        fwd = forward.isel(line=slice(0, num_pairs))
        bwd = backward.isel(line=slice(0, num_pairs))

        # Mask out zero rows (xarray preserves dims, so we need numpy for row-wise masking)
        fwd_vals = fwd.values
        bwd_vals = bwd.values

        mask = ~((fwd_vals == 0).all(axis=1) | (bwd_vals == 0).all(axis=1))
        fwd_vals = fwd_vals[mask]
        bwd_vals = bwd_vals[mask]

        # Subtract mean along each line (axis=1)
        fwd_vals -= fwd_vals.mean(axis=1, keepdims=True)
        bwd_vals -= bwd_vals.mean(axis=1, keepdims=True)

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


