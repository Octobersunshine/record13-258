import numpy as np
from typing import Tuple


def fft_spectrum(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n == 0:
        raise ValueError("Signal array must not be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    fft_result = np.fft.rfft(signal)
    magnitude = 2.0 * np.abs(fft_result) / n
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs, magnitude
