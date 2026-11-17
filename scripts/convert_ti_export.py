"""Convert a TI AFE4950 GUI export (as in HackEdge data/) into the data/own format.

TI export columns: ECG_Time, ECG_Value (V, 500 Hz), TIA1-3_Time, TIA1-3_Value (V, 50 Hz), ...
(trailing comma on every row). Output: time_s, ecg, ppg at the ECG rate; the optical
channel is linearly interpolated onto the ECG clock, or left blank if it is empty.

    python scripts/convert_ti_export.py RAW.csv --out data/own --subject nhat --session late \
        --kss 8                       # add --baseline for the rested/baseline recording

Appends (or updates) the row in data/own/sessions.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from breathbreak import CFG  # noqa: E402
from breathbreak.signal_quality import check_ecg, check_ppg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="data/own")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--kss", type=int, help="Karolinska Sleepiness Scale 1-9 at recording time")
    ap.add_argument("--baseline", action="store_true", help="rested recording used as personal baseline")
    a = ap.parse_args()

    df = pd.read_csv(a.src, index_col=False)
    t_ecg = df["ECG_Time"].dropna().to_numpy(float)
    ecg = df["ECG_Value"].dropna().to_numpy(float)[: len(t_ecg)]
    fs = round(1.0 / np.median(np.diff(t_ecg)))
    qe = check_ecg(ecg, fs, CFG)
    print(f"ECG check: {qe.message()}")
    if not qe.ok:
        ecg = np.full(len(t_ecg), np.nan)                 # never store unusable data as if it were real
    ppg = np.full(len(t_ecg), np.nan)
    if "TIA1-3_Value" in df:
        t_o = df["TIA1-3_Time"].dropna().to_numpy(float)
        o = df["TIA1-3_Value"].dropna().to_numpy(float)[: len(t_o)]
        if len(o) > 10:
            qp = check_ppg(o, round(1.0 / np.median(np.diff(t_o))), qe.metrics.get("hr_bpm") if qe.ok else None)
            print(f"PPG check: {qp.message()}")
            if qp.ok:
                ppg = np.interp(t_ecg, t_o, o, left=np.nan, right=np.nan)

    os.makedirs(a.out, exist_ok=True)
    name = f"{a.subject}_{a.session}.csv"
    pd.DataFrame({"time_s": t_ecg, "ecg": ecg, "ppg": ppg}).to_csv(os.path.join(a.out, name), index=False)

    ses_path = os.path.join(a.out, "sessions.csv")
    cols = ["subject", "session", "file", "fs", "kss", "is_baseline"]
    ses = pd.read_csv(ses_path) if os.path.exists(ses_path) else pd.DataFrame(columns=cols)
    ses = ses[~((ses.subject.astype(str) == a.subject) & (ses.session.astype(str) == a.session))]
    ses.loc[len(ses)] = [a.subject, a.session, name, fs, a.kss, int(a.baseline)]
    ses.to_csv(ses_path, index=False)
    dur = t_ecg[-1] - t_ecg[0]
    print(f"wrote {name}: {dur:.0f} s at {fs} Hz, optical {'present' if np.isfinite(ppg).any() else 'empty'}")
    if dur < 300:
        print(f"warning: {dur:.0f} s is short - record >= 5 min (2-min baseline + 60-s windows).")


if __name__ == "__main__":
    main()
