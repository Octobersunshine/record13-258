import numpy as np
from typing import Tuple


_WINDOW_ALIASES = {
    'hann': 'hanning',
    'rect': 'ones',
    'boxcar': 'ones',
    'rectangular': 'ones',
}

_WINDOW_FUNCS = {'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser'}


def _get_window(name: str, n: int) -> np.ndarray:
    name = _WINDOW_ALIASES.get(name, name)
    if name == 'kaiser':
        return np.kaiser(n, 14.0)
    if name in _WINDOW_FUNCS:
        return getattr(np, name)(n)
    if name == 'ones':
        return np.ones(n)
    raise ValueError(f"Unknown window: {name!r}")


def fft_spectrum(signal: np.ndarray, sample_rate: float, window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n == 0:
        raise ValueError("Signal array must not be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    win = _get_window(window, n) if window else np.ones(n)
    coherent_gain = win.sum() / n

    windowed = signal * win
    fft_result = np.fft.rfft(windowed)
    magnitude = 2.0 * np.abs(fft_result) / (n * coherent_gain)
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs, magnitude
