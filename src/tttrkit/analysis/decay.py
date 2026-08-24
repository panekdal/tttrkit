"""Operations on TCSPC decay histograms."""

import numpy as np


def shift_decay(arr: np.ndarray, n: int) -> np.ndarray:
    """Shift a decay histogram by ``n`` bins, wrapping at the boundaries."""
    return np.roll(arr, -n)