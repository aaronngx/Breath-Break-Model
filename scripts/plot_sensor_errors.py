"""Picture of the sensor problems in the HackEdge recordings (good vs rejected), to share with the team.

    python scripts/plot_sensor_errors.py data/hackedge      -> reports/sensor_errors.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.hackedge import _volts  # noqa: E402
from breathbreak.signal_quality import ECG_MIN_TEMPLATE_CORR, check_ecg, check_ppg  # noqa: E402

GOOD, BAD, INK, MUTED = "#2563c9", "#c2410c", "#16202c", "#7a8494"


def load(root, name, ch):
    df = pd.read_csv(os.path.join(root, name), index_col=False)
    t, v = _volts(df, ch)
    return t - t[0], v, round(1.0 / np.median(np.diff(t)))


def beat_corr(x, fs, peaks):
    rr = np.median(np.diff(peaks))
    pre, post = int(min(0.2 * fs, 0.3 * rr)), int(min(0.4 * fs, 0.6 * rr))
    ok = [p for p in peaks if p - pre >= 0 and p + post <= len(x)]
    beats = np.array([x[p - pre:p + post] for p in ok])
    tmpl = np.median(beats, axis=0)
    return np.array(ok), np.array([np.corrcoef(b, tmpl)[0, 1] for b in beats]), beats, tmpl, pre


def ecg_panel(ax, axb, root, name, title):
    _, v, fs = load(root, name, "ECG")
    q = check_ecg(v, fs, CFG)
    f = CFG.ecg_fs
    raw = S.resample(v, fs, f)
    peaks = S.ecg_rpeaks(raw, f)                       # same steps as check_ecg
    x = S.bandpass(raw, f, 0.5, 40) * 1000
    pk, corr, beats, tmpl, pre = beat_corr(x, f, peaks)
    t = np.arange(len(x)) / f
    w = (t >= 5) & (t < 15)
    ax.plot(t[w], x[w], color=INK, lw=0.8)
    for p, c in zip(pk, corr):
        if 5 <= p / f < 15:
            ax.plot(p / f, x[p], "o", ms=6, color=GOOD if c >= ECG_MIN_TEMPLATE_CORR else BAD,
                    mec="white", mew=1, zorder=3)
    status = "PASS" if q.ok else "REJECTED: " + ", ".join(q.issues)
    ax.set_title(f"{title}\n{name}  ·  {status}", loc="left", fontsize=10,
                 color=INK if q.ok else BAD)
    ax.set_xlabel("seconds"); ax.set_ylabel("ECG (mV)")
    tb = (np.arange(beats.shape[1]) - pre) / f * 1000
    for b, c in zip(beats, corr):
        axb.plot(tb, b, lw=0.7, alpha=0.6, color=GOOD if c >= ECG_MIN_TEMPLATE_CORR else BAD)
    axb.plot(tb, tmpl, color=INK, lw=2)
    good = np.mean(corr >= ECG_MIN_TEMPLATE_CORR)
    axb.set_title(f"every beat overlaid: {good:.0%} match the average\n(need ≥ 80 %)", loc="left",
                  fontsize=10, color=INK if good >= 0.8 else BAD)
    axb.set_xlabel("ms from R peak")


def ppg_panel(ax, root, name, title, ecg_hr):
    t, v, fs = load(root, name, "TIA1-3")
    q = check_ppg(v, fs, ecg_hr)
    w = (t >= 5) & (t < 15)
    y = (v[w] - np.median(v[w])) * 1000
    ax.plot(t[w], y, color=GOOD if q.ok else BAD, lw=1.2)
    status = "PASS" if q.ok else "REJECTED: " + ", ".join(q.issues)
    ax.set_title(f"{title}\n{name}  ·  {status}  ·  whole-file swing {q.metrics['ptp_mV']:.2f} mV",
                 loc="left", fontsize=10, color=INK if q.ok else BAD)
    ax.set_xlabel("seconds"); ax.set_ylabel("optical (mV, centred)")
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/hackedge")
    ap.add_argument("--out", default="reports/sensor_errors.png")
    a = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 0.05], hspace=0.55, wspace=0.25)

    ecg_panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), a.root,
              "fatigue/third_attempt.csv", "ECG  ·  GOOD (for comparison)")
    ecg_panel(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), a.root,
              "fatigue/fourth_attempt.csv", "ECG  ·  PROBLEM: beats change shape (movement / loose electrode)")

    ax_g = fig.add_subplot(gs[2, 0])
    sub = gs[2, 1].subgridspec(2, 1, hspace=0.9)
    ppg_panel(fig.add_subplot(sub[0]), a.root, "normal/first_attempt.csv", "PPG  ·  GOOD", None)
    yb = ppg_panel(fig.add_subplot(sub[1]), a.root, "fatigue/first_attempt.csv",
                   "PPG  ·  PROBLEM: no signal", None)
    # big panel: both optical traces on the same scale, so the flat one is obvious
    tg, vg, _ = load(a.root, "normal/first_attempt.csv", "TIA1-3")
    w = (tg >= 5) & (tg < 15)
    ax_g.plot(tg[w], (vg[w] - np.median(vg[w])) * 1000, color=GOOD, lw=1.2, label="normal/first_attempt (passes)")
    ax_g.plot(np.linspace(5, 15, len(yb)), yb, color=BAD, lw=1.5,
              label="fatigue/first_attempt (+ 7 other files): flat line")
    ax_g.set_title("PPG  ·  same scale: a real pulse vs the 8 rejected files\n"
                   "optical sensor not touching the skin, or the channel is switched off",
                   loc="left", fontsize=10, color=BAD)
    ax_g.set_xlabel("seconds"); ax_g.set_ylabel("optical (mV, centred)")
    ax_g.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("HackEdge recordings: where the sensor data is bad", x=0.06, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.935, "Blue dot = heartbeat matches the average beat.  Red = it does not.  "
             "Rejected channels are never fed to the model.  Rerun: python scripts/plot_sensor_errors.py data/hackedge",
             fontsize=9.5, color=MUTED)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
