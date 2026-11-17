"""Replay a recording through the realtime scorer, as the AM62 would run it live.

  python scripts/realtime_demo.py --model models/model_a.joblib --csv data/own/nhat_rested.csv data/own/nhat_late.csv --fs 500 --column ecg

For a live device, replace the replay loop with chunks read from the AFE4950
(e.g. every 1 s) and call scorer.push(chunk) the same way.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from breathbreak.realtime import RealtimeScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True, nargs="+", help="one or more CSVs played back to back (e.g. rested.csv late.csv)")
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--column", default="ecg")
    ap.add_argument("--chunk-s", type=float, default=1.0)
    a = ap.parse_args()

    x = pd.concat([pd.read_csv(c)[a.column] for c in a.csv]).to_numpy(float)
    scorer = RealtimeScorer.from_file(a.model, input_fs=a.fs)
    n = int(a.chunk_s * a.fs)
    for i in range(0, len(x), n):
        for ev in scorer.push(x[i:i + n]):
            t = ev["t_s"] / 60
            if ev["type"] == "window":
                bar = "#" * int(ev["p_smoothed"] * 20)
                print(f"{t:6.1f} min  HR {ev['hr']:5.1f}  RMSSD {ev['rmssd']:6.1f}  "
                      f"P(fatigued) {ev['p_smoothed']:.2f} {bar}")
            else:
                print(f"{t:6.1f} min  {ev['type']}  {ev.get('msg', '')}")


if __name__ == "__main__":
    main()
