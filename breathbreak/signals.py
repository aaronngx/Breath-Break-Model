"""Signal processing: filtering, resampling, beat detection, IBI cleaning."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample (keeps timing accurate, no FFT edge ringing)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    from fractions import Fraction
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(np.asarray(x, dtype=float), frac.numerator, frac.denominator)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------
def ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Return R-peak sample indices from an ECG segment."""
    ecg = np.asarray(ecg, dtype=float)
    if nk is not None:
        clean = nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit", correct_artifacts=False)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    # Fallback: band-pass + adaptive threshold
    f = bandpass(ecg, fs, 5, 20)
    energy = np.abs(f)
    peaks, _ = sps.find_peaks(energy, distance=int(0.3 * fs), height=np.percentile(energy, 90) * 0.5)
    return peaks


def ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Return systolic peak indices from a PPG segment.

    NOTE: some devices (e.g. Galaxy Watch) record inverted PPG. Pass -ppg if
    the pulse looks upside down.
    """
    ppg = np.asarray(ppg, dtype=float)
    if nk is not None:
        clean = nk.ppg_clean(ppg, sampling_rate=fs, method="elgendi")
        info = nk.ppg_findpeaks(clean, sampling_rate=fs, method="elgendi")
        return np.asarray(info["PPG_Peaks"], dtype=int)
    f = bandpass(ppg, fs, 0.5, 4)
    peaks, _ = sps.find_peaks(f, distance=int(0.33 * fs))
    return peaks


# ---------------------------------------------------------------------------
# Inter-beat intervals
# ---------------------------------------------------------------------------
def peaks_to_ibi_ms(peaks: np.ndarray, fs: float) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000.0


def clean_ibi(ibi_ms: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    """Remove physiologically impossible IBIs and ectopic/missed beats.

    Returns (clean_ibi, fraction_kept).
    """
    ibi = np.asarray(ibi_ms, dtype=float)
    if ibi.size == 0:
        return ibi, 0.0
    ok = (ibi >= cfg.ibi_min_ms) & (ibi <= cfg.ibi_max_ms)
    # local median filter (5 beats) to catch missed / extra beats
    if ibi.size >= 5:
        med = sps.medfilt(ibi, kernel_size=5)
        ok &= np.abs(ibi - med) <= cfg.ibi_max_jump * med
    return ibi[ok], float(ok.mean())


def window_ok(ibi_clean: np.ndarray, frac_kept: float, cfg) -> bool:
    return ibi_clean.size >= cfg.min_beats_per_window and frac_kept >= cfg.min_clean_fraction
