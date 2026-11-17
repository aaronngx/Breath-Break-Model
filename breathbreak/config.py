"""Central settings for the BreathBreak fatigue models.

Change values here instead of inside the code.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Windowing -------------------------------------------------------
    window_s: float = 60.0          # each decision covers 60 s (slide 3)
    step_s: float = 30.0            # 50% overlap for training windows
    baseline_s: float = 120.0       # 2-minute start-of-shift baseline

    # --- Sampling rates used internally -----------------------------------
    ecg_fs: int = 250               # all ECG is resampled to this rate
    ppg_fs: int = 64                # all PPG is resampled to this rate

    # --- Beat / IBI cleaning ---------------------------------------------
    ibi_min_ms: float = 300.0       # 200 bpm
    ibi_max_ms: float = 2000.0      # 30 bpm
    ibi_max_jump: float = 0.20      # drop IBIs >20% away from local median
    min_beats_per_window: int = 40  # reject window if fewer clean beats
    min_clean_fraction: float = 0.80

    # --- Labels ------------------------------------------------------------
    kss_fatigue_threshold: int = 7  # KSS >= 7 -> "fatigued" (DROZY / own data)
    cfs_fatigue_threshold: int = 12 # Chalder >= 12 -> "fatigued" (MEFAR)

    # --- Model A ---------------------------------------------------------
    # 60-s-valid, time-domain HRV only (Shaffer et al. 2020: LF/HF needs ~180 s)
    model_a_features: tuple = ("mean_hr", "sdnn", "rmssd", "pnn50", "cvnn")
    normalize_to_baseline: bool = True

    # --- Model B ---------------------------------------------------------
    embedder: str = "papagei"       # "papagei" | "pulseppg" | "prv" (no weights needed)
    embedder_weights: str = "weights/papagei_s.pt"
    embedder_repo: str = "third_party/papagei-foundation-model"

    # --- Realtime alerting ---------------------------------------------------
    alert_threshold: float = 0.6    # P(fatigued) above this counts as "high"
    alert_consecutive: int = 2      # need N high windows in a row to alert
    ema_alpha: float = 0.5          # smoothing of the probability

    random_state: int = 42
    extra: dict = field(default_factory=dict)


CFG = Config()
