"""BreathBreak - passive shift monitor.

    streamlit run app/breathbreak_app.py

The worker puts the band on at the start of the shift and takes it off at the end. No questions, no tests.
  Worker view   band check alerts, green / yellow / red, break suggestions with short recovery instructions,
                guided breathing, recovery check after each break, end-of-shift summary.
  HR view       breaks needed vs taken, time on shift, longest stretch without a break, signal quality.
                No heart readings and no overtime verdict (switched off until validated on pilot data).
  Pilot mode    adds a 1-5 "how tired are you" tap: the labels that will train and test the real model.
Logic: breathbreak/shift_monitor.py. Data stays in data/app/ on this machine (gitignored).
This version replays a recorded shift; live streaming from the band is not wired yet.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breathbreak import CFG  # noqa: E402
from breathbreak import shift_monitor as M  # noqa: E402
from breathbreak import wear_check as W  # noqa: E402
from breathbreak.data.hackedge import read_ti_export  # noqa: E402
from breathbreak.demo_shift import make_demo_shift  # noqa: E402
from breathbreak.fusion import load_fusion  # noqa: E402

APP_DIR = ROOT / "data" / "app"
SHIFTS = APP_DIR / "shifts.csv"
BASELINES = APP_DIR / "baselines.csv"
LABELS = APP_DIR / "pilot_labels.csv"
REC_DIR = APP_DIR / "recordings"
DEMO_DIR = ROOT / "data" / "hackedge_latest"

LEVEL = {0: ("You're doing fine", "Keep going. The band will tell you when to rest.", "#16a34a", "#dcfce7"),
         1: ("Getting tired", "A short break now keeps you sharp for the rest of the shift.", "#ca8a04", "#fef9c3"),
         2: ("Time to recover", "Your body needs a proper break before you carry on.", "#dc2626", "#fee2e2")}
PANEL = "background:#151b23;border-radius:18px;padding:18px 20px;font:14px/1.45 system-ui,Segoe UI,sans-serif;color:#e5e7eb"
LEVEL_COLOR = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}
TL_COLOR = {**LEVEL_COLOR, "break": "#3b82f6", "nosignal": "#9ca3af"}
TL_NAME = {0: "fine", 1: "getting tired", 2: "time to recover", "break": "break", "nosignal": "no signal"}
ISSUE_COLOR = {"loose": "#eab308", "too_tight": "#f97316", "no_contact": "#ef4444"}

st.set_page_config(page_title="BreathBreak", layout="centered")
st.markdown("""<style>
.block-container {max-width: 680px; padding-top: 2rem;}
.card {border-radius: 18px; padding: 20px 24px; margin: 8px 0 16px;}
.card h2 {margin: 0 0 6px; font-size: 1.8rem;}
.card p.why {margin: 0 0 10px; opacity: .85;}
.card li {margin: 4px 0; font-size: 1.05rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource
def schedule_model():
    return load_fusion(str(ROOT / "models" / "fatigue_schedule.joblib"))


def read_csv(p):
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def history_baseline(worker):
    b = read_csv(BASELINES)
    if b.empty or worker not in set(b.worker):
        return None
    b = b[b.worker == worker].tail(5)
    return {"hr": float(b.hr.median()), "rmssd": float(b.rmssd.median())}


@st.cache_data(show_spinner="Reading the band data...")
def analyse_demo(start, hist):
    x, fs, taps, _ = make_demo_shift(start)
    ep = M.epoch_table(x, fs)
    br = M.infer_breaks(ep, taps)
    return ep, M.monitor(ep, br, start, schedule_model(), hist)


@st.cache_data(show_spinner="Reading the band data...")
def analyse_file(path, start, hist):
    df = pd.read_csv(path, index_col=False)
    if "TIA1-3_Time" not in df:
        return None, None, None
    t, v = df["TIA1-3_Time"].to_numpy(float), df["TIA1-3_Value"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v)
    ppg, fs = v[ok], float(round(1.0 / np.median(np.diff(t[ok]))))
    sig = read_ti_export(str(path), CFG)
    ecg, fs_e = sig["ecg"] if sig["ecg"] else (None, None)
    fit = W.fit_check(ppg, fs).status
    ep = M.epoch_table(ppg, fs, ecg, fs_e)
    if ep.empty:
        return ep, None, fit
    br = M.infer_breaks(ep)
    return ep, M.monitor(ep, br, start, schedule_model(), hist), fit


# ------------------------------------------------------------------ pieces
def card(title, why, color, bg, steps=()):
    items = "".join(f"<li>{x}</li>" for x in steps)
    st.html(f"""<div style="background:{bg};border:2px solid {color};border-radius:18px;padding:18px 22px;
color:#111827;font:15px/1.5 system-ui,Segoe UI,sans-serif">
<div style="display:flex;align-items:center;gap:10px;font-size:26px;font-weight:700;margin-bottom:6px">
<span style="width:14px;height:14px;border-radius:50%;background:{color};flex:none"></span>{title}</div>
<div style="color:#374151">{why}</div>{f'<ol style="margin:10px 0 0;padding-left:22px">{items}</ol>' if items else ""}</div>""")


def ring(frac, big, unit, label, color):
    deg = 360 * float(min(max(frac, 0), 1))
    return f"""<div style="text-align:center;min-width:110px">
<div style="width:116px;height:116px;border-radius:50%;margin:0 auto;
background:conic-gradient({color} {deg:.0f}deg,#2b3440 0)">
<div style="position:relative;top:11px;left:11px;width:94px;height:94px;border-radius:50%;background:#151b23;
display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;color:#fff">
{big}<span style="font-size:14px;margin-left:2px">{unit}</span></div></div>
<div style="margin-top:8px;font-size:12px;font-weight:600;letter-spacing:.12em;color:#cbd5e1">{label}</div></div>"""


def rings(on_shift_h, energy, level, since_break_h):
    st.html(f"""<div style="display:flex;justify-content:space-around;flex-wrap:wrap;gap:8px;background:#151b23;
border-radius:18px;padding:20px 8px 16px;font-family:system-ui,Segoe UI,sans-serif">
{ring(on_shift_h / 12, f"{on_shift_h:.1f}", "h", "ON SHIFT", "#0ea5e9")}
{ring(energy / 100, energy, "%", "ENERGY", LEVEL_COLOR[level])}
{ring(since_break_h / 2, f"{since_break_h:.1f}", "h", "SINCE BREAK", "#f59e0b" if since_break_h >= 2 else "#93b4d6")}
</div>""")


def breathing():
    st.html("""<style>
@keyframes bb {0% {transform:scale(.55)} 40% {transform:scale(1)} 100% {transform:scale(.55)}}
@keyframes inw {0%,39% {opacity:1} 40%,100% {opacity:0}}
@keyframes outw {0%,39% {opacity:0} 40%,100% {opacity:1}}
.bwrap {display:flex; flex-direction:column; align-items:center; padding:10px 0 4px;
        font:600 18px system-ui,Segoe UI,sans-serif; color:#3b82f6;}
.bc {width:170px; height:170px; border-radius:50%; background:radial-gradient(#bfdbfe,#60a5fa);
     animation:bb 10s ease-in-out infinite;}
.bl {position:relative; height:28px; width:200px; text-align:center; margin-top:12px;}
.bl span {position:absolute; left:0; right:0;}
.bl .i {animation:inw 10s infinite} .bl .o {animation:outw 10s infinite}
</style><div class="bwrap"><div class="bc"></div>
<div class="bl"><span class="i">Breathe in - 4 seconds</span><span class="o">Breathe out - 6 seconds</span></div></div>""")
    st.caption("Follow the circle for 1-5 minutes. Slow breathing out longer than in helps the body settle.")


def clock(start, m):
    return f"{(start + timedelta(minutes=int(m))):%H:%M}"


def shift_panel(dec, breaks, upto, start, end_min):
    runs = []
    for _, r in dec[dec.t_min <= upto].iterrows():
        k = "break" if r.on_break else ("nosignal" if (not r.band_ok and r.band_issue) else int(r.level))
        if runs and runs[-1][2] == k:
            runs[-1][1] = r.t_min
        else:
            runs.append([r.t_min - M.DECIDE_EVERY_MIN, r.t_min, k])
    span = max(end_min, 1)
    bar = "".join(f'<div title="{clock(start, a)}-{clock(start, b)}: {TL_NAME[k]}" style="flex:{b - a};'
                  f'background:{TL_COLOR[k]}"></div>' for a, b, k in runs)
    if end_min > upto:
        bar += f'<div style="flex:{end_min - upto};background:#2b3440"></div>'
    tri = "width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:8px solid #f8fafc"
    marks = "".join(f'<div title="break suggested {clock(start, m)}" style="position:absolute;left:{100 * m / span:.2f}%;'
                    f'top:0;transform:translateX(-5px);{tri}"></div>'
                    for m in dec[(dec.t_min <= upto) & dec.prompt.notna()].t_min)
    legend = "".join(f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:10px;'
                     f'height:10px;border-radius:2px;background:{TL_COLOR[k]};margin-right:5px"></span>{TL_NAME[k]}</span>'
                     for k in TL_COLOR)
    legend += (f'<span style="white-space:nowrap"><span style="display:inline-block;{tri};margin-right:5px"></span>'
               'break suggested</span>')
    rows = ""
    for b in [b for b in breaks if b.end_min <= upto]:
        rec = b.recovery
        if rec.get("measured"):
            ok = rec["recovered"]
            res = (f'<span style="color:{"#4ade80" if ok else "#facc15"}">{"Recovered" if ok else "Partly recovered"}</span>'
                   f'<span style="color:#94a3b8"> - heart rate {rec["hr_change_bpm"]:+.0f}, HRV {rec["hrv_change_pct"]:+d}%</span>')
        else:
            res = '<span style="color:#94a3b8">Band was off - recovery not measured</span>'
        rows += (f'<div style="display:flex;gap:14px;padding:7px 0;border-top:1px solid #2b3440">'
                 f'<span style="width:96px;flex:none;font-variant-numeric:tabular-nums">{clock(start, b.start_min)}-'
                 f'{clock(start, b.end_min)}</span><span>{res}</span></div>')
    rows = rows or '<div style="padding:7px 0;border-top:1px solid #2b3440;color:#94a3b8">No breaks yet.</div>'
    st.html(f"""<div style="{PANEL}">
<div style="position:relative;padding-top:10px"><div style="display:flex;height:26px;border-radius:6px;overflow:hidden">{bar}</div>
<div style="position:absolute;left:0;right:0;top:0">{marks}</div></div>
<div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:12px;margin-top:4px">
<span>{clock(start, 0)}</span><span>{clock(start, end_min)}</span></div>
<div style="margin:8px 0 14px;font-size:12px;color:#cbd5e1">{legend}</div>
<div style="font-weight:600;margin-bottom:2px">Breaks</div>{rows}</div>""")


def tiles(items):
    cells = "".join(f'<div style="background:#1f2733;border-radius:12px;padding:12px 14px"><div style="font-size:12px;'
                    f'color:#94a3b8">{k}</div><div style="font-size:22px;font-weight:700;margin-top:2px">{v}</div></div>'
                    for k, v in items)
    st.html(f"""<div style="{PANEL};display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">
{cells}</div>""")


def hm(minutes):
    return f"{int(minutes // 60)}h {int(minutes % 60):02d}m"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    view = st.radio("View", ["Worker", "HR dashboard"])
    worker = st.text_input("Worker ID", value="W01").strip() or "W01"
    st.divider()
    src = st.radio("Band data", ["Simulated 12-hour shift (demo)", "Recorded file"])
    pilot = st.toggle("Pilot mode", help="Adds a 1-5 tiredness tap. Used only while collecting data to train the model.")


# ------------------------------------------------------------------ HR view
def hr_view():
    st.title("BreathBreak - team view")
    st.caption("Breaks and time on shift only. Heart readings stay with the worker.")
    sh = read_csv(SHIFTS)
    if sh.empty:
        st.info("No saved shifts yet. Finish a shift in the Worker view and press Save shift.")
        return
    if (sh.source == "demo").any():
        if not st.toggle("Include demo shifts", value=True):
            sh = sh[sh.source != "demo"]
        else:
            st.caption("Some rows come from the simulated demo shift, not real workers.")
    if sh.empty:
        st.info("No real shifts saved yet.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shifts", len(sh))
    c2.metric("Avg hours worn", f"{sh.hours_worn.mean():.1f}")
    c3.metric("Breaks suggested", int(sh.breaks_suggested.sum()))
    c4.metric("Breaks taken", int(sh.breaks_taken.sum()))

    g = sh.groupby("worker").agg(shifts=("shift_id", "count"), hours=("hours_worn", "mean"),
                                 suggested=("breaks_suggested", "mean"), taken=("breaks_taken", "mean"),
                                 longest_no_break=("longest_without_break_h", "max"),
                                 red_min=("min_red", "mean"), signal=("good_signal_pct", "mean"),
                                 last=("shift_start", "max")).reset_index()

    def flags(r):
        f = []
        if r.longest_no_break >= 3:
            f.append("long stretches without a break")
        if r.red_min >= 60:
            f.append("often needs recovery")
        if r.taken < 0.5 * r.suggested:
            f.append("skips suggested breaks")
        if r.signal < 70:
            f.append("band fit - check signal")
        return ", ".join(f) or "-"

    g["attention"] = g.apply(flags, axis=1)
    st.subheader("Workers")
    st.dataframe(g.rename(columns={"hours": "avg hours", "suggested": "breaks suggested / shift",
                                   "taken": "breaks taken / shift", "longest_no_break": "longest without break (h)",
                                   "red_min": "min needing recovery / shift", "signal": "good signal %",
                                   "last": "last shift"}).round(1), hide_index=True, width="stretch")
    st.subheader("Shifts")
    cols = ["worker", "shift_start", "hours_worn", "breaks_suggested", "breaks_taken", "breaks_after_suggestion",
            "longest_without_break_h", "min_yellow", "min_red", "good_signal_pct", "source"]
    st.dataframe(sh.reindex(columns=cols).sort_values("shift_start", ascending=False), hide_index=True, width="stretch")
    st.info("Overtime advice is switched off until the model has been validated on pilot data. These numbers describe "
            "breaks and time worked - they do not say whether someone is fit for overtime.")


# ------------------------------------------------------------------ worker view
def worker_view():
    st.title("BreathBreak")
    today = datetime.now().replace(second=0, microsecond=0)
    hist = history_baseline(worker)
    if src.startswith("Simulated"):
        start = today.replace(hour=7, minute=0)
        ep, run = analyse_demo(start, hist)
        source = "demo"
        st.caption(f"Worker {worker} - band on at {start:%H:%M}. Made-up shift to show how the app behaves, not real data.")
    else:
        files = sorted(DEMO_DIR.glob("*/*.csv")) if DEMO_DIR.exists() else []
        up = st.file_uploader("Band recording (.csv)", type="csv")
        path = None
        if up:
            REC_DIR.mkdir(parents=True, exist_ok=True)
            path = REC_DIR / f"{worker}_{today:%Y%m%d_%H%M}_{Path(up.name).name}"
            if not path.exists():
                path.write_bytes(up.getvalue())
        elif files:
            path = st.selectbox("Or an example recording", files,
                                format_func=lambda p: f"{p.parent.name} / {p.stem.replace('_', ' ')}")
        if path is None:
            st.info("Upload a recording from the band.")
            return
        t0 = st.time_input("Band put on at", value=today.replace(hour=7, minute=0).time(), step=900)
        start = today.replace(hour=t0.hour, minute=t0.minute)
        ep, run, fit = analyse_file(str(path), start, hist)
        source = "file"
        if ep is None:
            st.error("This file has no optical (PPG) channel.")
            return
        if run is None or run["decisions"].empty:
            st.warning(f"This recording is too short for the shift monitor (needs at least {M.DECIDE_EVERY_MIN} "
                       "minutes; a real shift is 10-12 hours). Showing the band check only.")
            if fit != "good":
                title, why, steps = W.FIX[fit]
                card("Check your band: " + title.lower(), why, ISSUE_COLOR[fit], "#fef9c3", steps)
            else:
                st.success("The band is reading your pulse.")
            return

    dec, breaks, end_min = run["decisions"], run["breaks"], run["end_min"]
    T = st.select_slider("Replay the shift", options=list(dec.t_min), value=int(dec.t_min.iloc[-1]),
                         format_func=lambda m: f"{(start + timedelta(minutes=int(m))):%H:%M}")
    st.caption("On the band this runs live. Here you can move through a recorded shift.")
    r = dec[dec.t_min == T].iloc[0]

    # band fit
    if r.band_issue and not r.on_break:
        title, why, steps = W.FIX[r.band_issue]
        card("Check your band: " + title.lower(), why, ISSUE_COLOR[r.band_issue], "#fef9c3", steps)

    # main message
    past = dec[(dec.t_min <= T) & (dec.t_min > T - 15) & dec.prompt.notna()]
    br_now = next((b for b in breaks if b.start_min < T <= b.end_min), None)
    if br_now:
        card("On a break", f"Started {(start + timedelta(minutes=br_now.start_min)):%H:%M} - "
             f"{T - br_now.start_min} minutes so far. Relax, you'll get a reminder when it's time to go back.",
             "#3b82f6", "#dbeafe")
        breathing()
    elif len(past) and not any(past.t_min.iloc[-1] <= b.start_min <= T for b in breaks):
        p = past.iloc[-1]
        kind_title, mins, steps = M.BREAK_KINDS[p.prompt]
        title, _, color, bg = LEVEL[int(p.level)]
        card(f"{title}: take a {kind_title}", " ".join(p.reasons), color, bg, steps)
        if st.toggle("Start guided breathing"):
            breathing()
    else:
        title, why, color, bg = LEVEL[int(r.level)]
        card(title, " ".join(r.reasons) or why, color, bg)

    rings(T / 60, round(100 * (1 - r.risk)), int(r.level), r.since_break_min / 60)
    base = run["baseline"]
    if base is None:
        st.caption("Learning your normal: the band needs your first 30 minutes of good signal.")
    elif T <= M.BASELINE_MIN:
        st.caption("Learning your normal from the first 30 minutes of your shift.")
    else:
        st.caption(f"Your normal ({base['from']}): heart rate about {base['hr']:.0f}. "
                   "Energy combines time on shift, time of day and how your heart compares with your normal.")

    st.subheader("Your shift")
    shift_panel(dec, breaks, T, start, end_min)

    if pilot:
        st.subheader("Pilot: how tired are you right now?")
        st.caption("1 = wide awake, 5 = fighting sleep. Only during the pilot - this trains the model.")
        cols = st.columns(5)
        for i, c in enumerate(cols, start=1):
            if c.button(str(i), key=f"kss{i}", width="stretch"):
                APP_DIR.mkdir(parents=True, exist_ok=True)
                row = {"worker": worker, "shift_id": f"{worker}_{start:%Y%m%d_%H%M}", "t_min": int(T),
                       "time": (start + timedelta(minutes=int(T))).isoformat(timespec="minutes"), "rating": i,
                       "source": source, "level_shown": int(r.level), "risk": r.risk}
                pd.concat([read_csv(LABELS), pd.DataFrame([row])], ignore_index=True).to_csv(LABELS, index=False)
                st.success(f"Saved rating {i} at {row['time'][-5:]}.")

    if T == int(dec.t_min.iloc[-1]):
        s = run["summary"]
        st.subheader("End of shift")
        tiles([("Worn for", hm(s["hours_worn"] * 60)), ("Good signal", f"{s['good_signal_pct']}%"),
               ("Breaks suggested", s["breaks_suggested"]), ("Breaks taken", s["breaks_taken"]),
               ("Longest without a break", hm(s["longest_without_break_h"] * 60)),
               ("Time needing recovery", hm(s["min_red"]))])
        if st.button("Save shift", type="primary", width="stretch"):
            APP_DIR.mkdir(parents=True, exist_ok=True)
            sid = f"{worker}_{start:%Y%m%d_%H%M}"
            sh = read_csv(SHIFTS)
            if not sh.empty:
                sh = sh[sh.shift_id != sid]
            row = {"shift_id": sid, "worker": worker, "source": source, **s}
            pd.concat([sh, pd.DataFrame([row])], ignore_index=True).to_csv(SHIFTS, index=False)
            if base is not None and base["from"].startswith("first"):
                b = read_csv(BASELINES)
                if not b.empty:
                    b = b[b.shift_id != sid]
                pd.concat([b, pd.DataFrame([{"worker": worker, "shift_id": sid, "hr": round(base["hr"], 1),
                                             "rmssd": round(base["rmssd"], 1), "source": source}])],
                          ignore_index=True).to_csv(BASELINES, index=False)
            st.success("Shift saved. HR sees the break and time numbers, not your heart readings.")

    with st.expander("How this works"):
        st.markdown(f"""
- **Band check** every minute: is the sensor reading a pulse? If not, it tells you what to adjust.
- **Your normal** = your first {M.BASELINE_MIN} minutes of the shift (or your past shifts if the start was unreadable).
- **Every {M.DECIDE_EVERY_MIN} minutes** the app combines:
  - time on shift and time of day - learned from a sleep study (tested: AUC 0.79). It assumes you woke about
    {M.ASSUMED_WAKE_BEFORE_H:g} hour before your shift after {M.ASSUMED_SLEEP_H:g} hours of sleep;
  - how far your resting heart rate and heart-rate variability have moved from your normal - **not yet validated**,
    the pilot data will test and re-tune it;
  - {M.BREAK_DUE_MIN // 60} hours without a break - a rule from workplace safety research.
- A break is suggested only if tiredness lasts {M.PERSIST_MIN} minutes, at most once an hour unless it gets worse.
- **Recovery check**: heart rate and heart-rate variability before vs at the end of each break.
""")


if view == "Worker":
    worker_view()
else:
    hr_view()
