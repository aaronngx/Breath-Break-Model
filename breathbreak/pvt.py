"""Psychomotor Vigilance Test (PVT): scoring + a terminal runner for the SK-AM62.

A short PVT (2-3 min) measures sleepiness through attention lapses, which is
what PERCLOS and other drowsiness measures were validated against
(Dinges et al., FHWA/NHTSA 1998). Unlike heart-rate variability it does not
rise with stress.

Run on the board:   python -m breathbreak.pvt --minutes 3
"""
from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

LAPSE_MS = 500.0          # standard PVT lapse threshold
FALSE_START_MS = 100.0    # faster than this is anticipation, not a reaction
TIMEOUT_MS = 30_000.0     # no response -> counted as a (long) lapse

PVT_FEATURES = ("pvt_lapses", "pvt_speed")


def score_pvt(rt_ms, false_starts: int = 0) -> dict:
    """Summary metrics from a list of reaction times in milliseconds."""
    rt = np.asarray(rt_ms, dtype=float)
    early = rt < FALSE_START_MS
    valid = rt[~early]
    return {
        "pvt_n": int(valid.size),
        "pvt_lapses": int((valid > LAPSE_MS).sum()),
        "pvt_speed": float(np.mean(1000.0 / valid)) if valid.size else np.nan,   # responses per second
        "pvt_median_rt": float(np.median(valid)) if valid.size else np.nan,
        "pvt_false_starts": int(early.sum()) + int(false_starts),
    }


# ---------------------------------------------------------------------------
# Terminal runner (works over SSH / serial console; any key responds)
# ---------------------------------------------------------------------------
class _Keys:
    """Non-blocking single-key reads on Linux (the board) and Windows (dev PC)."""

    def __enter__(self):
        if sys.platform == "win32":
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32":
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def pressed(self) -> bool:
        if sys.platform == "win32":
            hit = False
            while self._msvcrt.kbhit():
                self._msvcrt.getwch()
                hit = True
            return hit
        import os
        import select
        hit = False
        while select.select([sys.stdin], [], [], 0)[0]:
            os.read(self._fd, 64)
            hit = True
        return hit


def run_pvt(minutes: float = 3.0, isi_s=(2.0, 10.0), seed: int | None = None) -> dict:
    """Interactive PVT. Press any key as soon as the counter appears."""
    rng = random.Random(seed)
    rts, false_starts = [], 0
    end = time.perf_counter() + minutes * 60
    print(f"PVT: {minutes:g} min. Press any key as soon as you see the counter. Get ready...")
    with _Keys() as keys:
        time.sleep(2.0)
        keys.pressed()
        while time.perf_counter() < end:
            wait_until = time.perf_counter() + rng.uniform(*isi_s)
            while time.perf_counter() < wait_until:
                if keys.pressed():
                    false_starts += 1
                    print("\r  too early!          ", end="", flush=True)
                time.sleep(0.002)
            t0 = time.perf_counter()
            while True:
                ms = (time.perf_counter() - t0) * 1000
                print(f"\r  >>> {ms:5.0f} ms <<<   ", end="", flush=True)
                if keys.pressed() or ms >= TIMEOUT_MS:
                    break
                time.sleep(0.001)
            rts.append(ms)
            print(f"\r  {ms:5.0f} ms           ", end="", flush=True)
    print()
    res = score_pvt(rts, false_starts)
    res["rts_ms"] = [round(r, 1) for r in rts]
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run a short PVT and print the score as JSON")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--out", help="optional JSON file to save the result")
    a = ap.parse_args()
    r = run_pvt(a.minutes)
    print(json.dumps({k: v for k, v in r.items() if k != "rts_ms"}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
