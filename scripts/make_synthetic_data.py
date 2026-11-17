"""Generate FAKE data in the same file formats as the real datasets, so you can
run every script end to end before the real downloads arrive.

  python scripts/make_synthetic_data.py      # writes data_synth/

Numbers produced on synthetic data mean NOTHING about real fatigue.
"""
import os
import pickle

import neurokit2 as nk
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
OUT = "data_synth"


def ecg(dur, fs, hr, hr_std, seed):
    return nk.ecg_simulate(duration=dur, sampling_rate=fs, heart_rate=hr, heart_rate_std=hr_std,
                           noise=0.02, random_state=seed)


def ppg(dur, fs, hr, rand, seed):
    return nk.ppg_simulate(duration=dur, sampling_rate=fs, heart_rate=hr, ibi_randomness=rand,
                           drift=0.1, motion_amplitude=0.05, random_state=seed)


def own(n_subj=8, dur=360, fs=250):
    d = os.path.join(OUT, "own"); os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(n_subj):
        hr0 = rng.uniform(62, 80)
        for sess, kss, dhr, std, rnd, base in [("rested", 3, 0, 3, 0.08, 1), ("late", 8, -6, 7, 0.18, 0)]:
            seed = i * 10 + base
            df = pd.DataFrame({"time_s": np.arange(dur * fs) / fs,
                               "ecg": ecg(dur, fs, hr0 + dhr, std, seed),
                               "ppg": ppg(dur, fs, hr0 + dhr, rnd, seed)})
            f = f"p{i}_{sess}.csv"; df.to_csv(os.path.join(d, f), index=False)
            rows.append({"subject": f"p{i}", "session": sess, "file": f, "fs": fs, "kss": kss, "is_baseline": base})
    pd.DataFrame(rows).to_csv(os.path.join(d, "sessions.csv"), index=False)


def mefar(n_subj=8, dur=360, fs=64):
    root = os.path.join(OUT, "MEFAR_raw"); rows = []
    for i in range(n_subj):
        hr0 = rng.uniform(62, 80)
        for sess, cfs, dhr, rnd in [("morning", 8, 0, 0.08), ("evening", 16, -5, 0.18)]:
            d = os.path.join(root, f"{i+1:02d}", sess); os.makedirs(d, exist_ok=True)
            x = ppg(dur, fs, hr0 + dhr, rnd, i * 7 + len(sess))
            np.savetxt(os.path.join(d, "BVP.csv"), np.r_[1.6e9, fs, x], fmt="%.6f")  # E4 format
            rows.append({"subject": f"{i+1:02d}", "session": sess, "cfs": cfs})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mefar_labels.csv"), index=False)


def wesad(n_subj=3, seg=300):
    for i in range(n_subj):
        hr0 = rng.uniform(65, 80)
        e = np.r_[ecg(seg, 700, hr0, 3, i), ecg(seg, 700, hr0 + 15, 2, i + 50)]   # stress: HR up, HRV down
        b = np.r_[ppg(seg, 64, hr0, 0.08, i), ppg(seg, 64, hr0 + 15, 0.05, i + 50)]
        lab = np.r_[np.ones(seg * 700), 2 * np.ones(seg * 700)].astype(int)
        d = os.path.join(OUT, "WESAD", f"S{i+2}"); os.makedirs(d, exist_ok=True)
        data = {"signal": {"chest": {"ECG": e[:, None]}, "wrist": {"BVP": b[:, None]}}, "label": lab}
        with open(os.path.join(d, f"S{i+2}.pkl"), "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    own(); mefar(); wesad()
    print(f"Synthetic data written to {OUT}/  (fake - for testing the pipeline only)")
