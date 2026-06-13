import numpy as np
from typing import Tuple, List, Optional


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


def find_top_peaks(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    top_n: int = 3,
    min_distance_hz: float = 1.0,
    min_magnitude: Optional[float] = None,
    skip_dc: bool = True,
) -> List[Tuple[float, float]]:
    if len(freqs) != len(magnitude):
        raise ValueError("freqs and magnitude must have the same length")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    start_idx = 1 if skip_dc else 0
    if start_idx >= len(magnitude) - 1:
        return []

    mag = magnitude[start_idx:]
    f = freqs[start_idx:]

    if min_distance_hz <= 0:
        min_bins = 1
    else:
        df = f[1] - f[0] if len(f) > 1 else 1.0
        min_bins = max(1, int(np.ceil(min_distance_hz / df)))

    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] <= mag[i - 1] or mag[i] <= mag[i + 1]:
            continue
        if min_magnitude is not None and mag[i] < min_magnitude:
            continue
        peaks.append((i, mag[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for idx, val in peaks:
        too_close = False
        for s_idx, _ in selected:
            if abs(idx - s_idx) < min_bins:
                too_close = True
                break
        if not too_close:
            selected.append((idx, val))
            if len(selected) >= top_n:
                break

    return [(float(f[idx]), float(val)) for idx, val in selected]


def fft_top_peaks(
    signal: np.ndarray,
    sample_rate: float,
    top_n: int = 3,
    window: str = 'hann',
    min_distance_hz: float = 1.0,
    min_magnitude: Optional[float] = None,
    skip_dc: bool = True,
) -> List[Tuple[float, float]]:
    freqs, mag = fft_spectrum(signal, sample_rate, window=window)
    return find_top_peaks(
        freqs, mag,
        top_n=top_n,
        min_distance_hz=min_distance_hz,
        min_magnitude=min_magnitude,
        skip_dc=skip_dc,
    )
