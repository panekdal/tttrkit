"""Frequency-domain analysis of FLIM decays and images."""

import numpy as np
from scipy.signal import convolve2d


def smooth_phasor(phasor, count, size: int = 3):
    kernel = np.ones((size, size), dtype=np.float32)
    valid = np.isfinite(phasor) & (count > 0)
    phasor_weighted = np.zeros_like(phasor, dtype=np.complex64)
    phasor_weighted[valid] = phasor[valid] * count[valid]
    count_weighted = np.zeros_like(count, dtype=np.float32)
    count_weighted[valid] = count[valid]
    num = convolve2d(
        phasor_weighted.real, kernel, mode="same"
    ) + 1j * convolve2d(phasor_weighted.imag, kernel, mode="same")
    den = convolve2d(count_weighted, kernel, mode="same")
    phasor_smoothed = np.full_like(phasor, np.nan + 1j * np.nan)
    mask = den > 0
    phasor_smoothed[mask] = num[mask] / den[mask]
    return phasor_smoothed


def get_phasor_from_decay(
    decay: np.ndarray,
    tcspc_resolution: float,
    sync_rate: float,
) -> complex:
    """Return the photon-weighted complex phasor of a TCSPC decay."""
    if decay.ndim != 1:
        raise ValueError("`decay` must be 1-D")
    total = decay.sum()
    if total == 0:
        return np.nan + 1j * np.nan
    time = np.arange(decay.size) * tcspc_resolution
    omega = 2 * np.pi * sync_rate
    return np.dot(decay, np.exp(1j * omega * time)) / total


def average_phasor(
    phasor: np.ndarray,
    photon_count: np.ndarray,
    mask: np.ndarray | None = None,
) -> complex:
    """Return the photon-weighted mean phasor over an optional ROI."""
    if phasor.shape != photon_count.shape:
        raise ValueError("phasor and photon_count must have identical shapes")
    if mask is not None and mask.shape != phasor.shape:
        raise ValueError("mask must have the same shape as phasor")
    valid = (photon_count > 0) & np.isfinite(phasor)
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.any(valid):
        return np.nan + 1j * np.nan
    total_photons = np.sum(photon_count[valid])
    return np.sum(phasor[valid] * photon_count[valid]) / total_photons