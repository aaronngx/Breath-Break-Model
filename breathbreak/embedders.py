"""PPG embedders for Model B (transfer learning, frozen backbone).

Every embedder maps raw PPG windows -> fixed-length vectors:
    emb = embedder.embed(X_windows, fs)      # X: [n_windows, n_samples]

Available:
  "prv"      - no pretrained weights; PPG pulse intervals -> HRV features.
               Use as a baseline and for testing the pipeline.
  "papagei"  - PaPaGei-S (Nokia Bell Labs, open weights). Recommended to start:
               its loading API is documented in the README.
  "pulseppg" - Pulse-PPG (field-trained, best fit for wrist PPG). Its README does
               not document an embedding API, so you must fill in `_load()` /
               `_forward()` after reading the repo's Colab demo.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from . import features as F
from . import signals as S


class PRVEmbedder:
    """PPG -> pulse-rate-variability features (6-dim). No deep learning."""
    name = "prv"

    def __init__(self, cfg):
        self.cfg = cfg

    def embed(self, X: np.ndarray, fs: float) -> np.ndarray:
        out = []
        for seg in X:
            ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(S.ppg_peaks(seg, fs), fs), self.cfg)
            if ibi.size < 5:
                out.append([np.nan] * len(F.FEATURE_NAMES))
            else:
                f = F.hrv_features(ibi)
                out.append([f[k] for k in F.FEATURE_NAMES])
        return np.asarray(out, dtype=float)


class PaPaGeiEmbedder:
    """Frozen PaPaGei-S encoder.

    Setup (once):
        git clone https://github.com/nokia-bell-labs/papagei-foundation-model third_party/papagei-foundation-model
        pip install -r third_party/papagei-foundation-model/requirements.txt
        # download papagei_s.pt from Zenodo record 13983110 into weights/

    PaPaGei expects 10-s segments at 125 Hz. Each 60-s window is split into
    six 10-s segments; their embeddings are mean-pooled.
    VERIFY `MODEL_CONFIG` against the README's quick-start before trusting results.
    """
    name = "papagei"
    FS = 125
    SEG_S = 10
    MODEL_CONFIG = dict(base_filters=32, kernel_size=3, stride=2, groups=1,
                        n_block=18, n_classes=512, n_experts=3)

    def __init__(self, cfg, device: str | None = None):
        import torch
        self.torch = torch
        self.cfg = cfg
        repo = os.path.abspath(cfg.embedder_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.resnet import ResNet1DMoE  # from the PaPaGei repo
        c = self.MODEL_CONFIG
        self.model = ResNet1DMoE(in_channels=1, base_filters=c["base_filters"], kernel_size=c["kernel_size"],
                                 stride=c["stride"], groups=c["groups"], n_block=c["n_block"],
                                 n_classes=c["n_classes"], n_experts=c["n_experts"])
        state = torch.load(cfg.embedder_weights, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _segments(self, seg: np.ndarray, fs: float) -> np.ndarray:
        x = S.resample(seg, fs, self.FS)
        x = S.bandpass(x, self.FS, 0.5, 12)
        n = self.FS * self.SEG_S
        segs = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
        segs = [(s - s.mean()) / (s.std() + 1e-8) for s in segs]
        return np.stack(segs)

    def embed(self, X: np.ndarray, fs: float, batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for seg in X:
                s = torch.tensor(self._segments(seg, fs)[:, None, :], dtype=torch.float32, device=self.device)
                e = self.model(s)
                e = e[0] if isinstance(e, (tuple, list)) else e   # PaPaGei-S returns (emb, experts, gates)
                out.append(e.mean(0).cpu().numpy())
        return np.stack(out)


class PulsePPGEmbedder:
    """Adapter for Pulse-PPG (https://github.com/maxxu05/pulseppg).

    Fill in `_load` and `_forward` after reading the repo's Colab demo
    (linked in its README). Everything else in Model B works unchanged.
    """
    name = "pulseppg"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self._load()

    def _load(self):
        raise NotImplementedError(
            "Pulse-PPG loading is not documented in its README. Open the Colab demo "
            "linked there, copy the model-construction + checkpoint-loading lines into "
            "PulsePPGEmbedder._load(), and the forward/pooling lines into _forward().")

    def _forward(self, x):  # x: np.ndarray [n_samples] at the model's expected fs
        raise NotImplementedError

    def embed(self, X: np.ndarray, fs: float) -> np.ndarray:
        return np.stack([self._forward(seg) for seg in X])


def get_embedder(cfg):
    return {"prv": PRVEmbedder, "papagei": PaPaGeiEmbedder, "pulseppg": PulsePPGEmbedder}[cfg.embedder](cfg)
