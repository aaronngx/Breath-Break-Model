"""BreathBreak - personal fatigue check (for the person wearing the sensor, not for managers).

    streamlit run app/breathbreak_app.py

1. Band check: is the wrist sensor reading a pulse? If not, what to adjust (breathbreak/wear_check.py).
2. Questions: wake time, sleep, shift start, last break, how you feel, quick taps (unwell / coffee / heat / water).
3. One answer (green / yellow / red) + what to do:
  - learned: schedule model (wake time, sleep, clock; DROZY LOSO AUC 0.79) and how you feel (KSS)
  - rules (no data to learn them yet): >= 2 h since last break, unwell, heart under more load than
    your own normal -> at least yellow. Coffee / heat / water only add notes.
Heart data never makes the answer red. HRV only from >= 2-min ECG recordings; "your normal" needs 3 calm checks.
Every saved check goes to data/app/log.csv (labelled sample for the model). Stays on this machine.
"""
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breathbreak import CFG  # noqa: E402
from breathbreak import features as F  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.hackedge import read_ti_export  # noqa: E402
from breathbreak.fusion import RETEST_RMSSD_DROP, load_fusion, predict_fatigue  # noqa: E402
from breathbreak.schedule import schedule_features  # noqa: E402
from breathbreak import wear_check as W  # noqa: E402

APP_DIR = ROOT / "data" / "app"
REC_DIR = APP_DIR / "recordings"
LOG = APP_DIR / "log.csv"
DEMO_DIR = ROOT / "data" / "hackedge_latest"
MIN_BASELINES = 3          # calm checks needed before "your normal" is used
MIN_HRV_S = 120            # HRV only from >= 2-min recordings (30-s clips vary 10-30 % by chance)
BREAK_DUE_H = 2.0          # accident risk climbs with time since the last break (Tucker et al. 2003)
LONG_SHIFT_H = 10.0

# simple 5-step feeling scale, stored as Karolinska Sleepiness Scale values
FEEL = {"Wide awake": 2, "Fine": 4, "A bit tired": 6, "Sleepy": 7, "Fighting sleep": 9}

LEVELS = {
    0: ("", "You're good", "#16a34a", "#dcfce7",
        ["Keep working.", "Stand up and stretch for 5 minutes every hour or so."]),
    1: ("", "Take a short break", "#ca8a04", "#fef9c3",
        ["Take **5–10 minutes** away from the screen: walk, drink water, breathe slowly.",
         "You'll feel a bit fresher, but it won't replace sleep.",
         "Check again in about an hour."]),
    2: ("", "Take a 10-minute nap", "#dc2626", "#fee2e2",
        ["A **10-minute nap** is the best fix — it helps for about 2–3 hours.",
         "No place to nap? Have a **coffee, then rest 15 minutes** with eyes closed.",
         "A walk alone won't fix this kind of tiredness.",
         "**Don't drive** or run machines until you've rested."]),
}

st.set_page_config(page_title="BreathBreak", layout="centered")
st.markdown("""<style>
.block-container {max-width: 640px; padding-top: 2rem;}
.card {border-radius: 18px; padding: 22px 24px; margin: 8px 0 18px;}
.card h2 {margin: 0 0 6px; font-size: 1.9rem;}
.card p.why {margin: 0 0 12px; opacity: .8;}
.card li {margin: 4px 0; font-size: 1.05rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def schedule_model():
    return load_fusion(str(ROOT / "models" / "fatigue_schedule.joblib"))


def load_log():
    return pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(
        columns=["file", "person", "recorded_at", "wake", "slept", "kss", "risk", "hr", "rmssd",
                 "ecg_ok", "ppg_ok", "condition"])


def hrv_from_ecg(ecg, fs):
    x = S.resample(np.asarray(ecg, float), fs, CFG.ecg_fs)
    pk = S.ecg_rpeaks(x, CFG.ecg_fs) / CFG.ecg_fs
    ibi, frac = S.clean_ibi(np.diff(pk) * 1000.0, CFG)
    if ibi.size < 15 or frac < 0.8:
        return None
    h = F.hrv_features(ibi)
    return {"hr": float(h["mean_hr"]), "rmssd": float(h["rmssd"])}


def heart_vs_normal(log, person, hr, rmssd):
    """Compare today with this person's own calm checks (KSS <= 4). Flags only changes bigger than
    their usual day-to-day spread (1.5 SD), with a floor so tiny spreads don't make it jumpy.
    Returns True if the heart looks 'under more load than usual'. Shown only to the wearer."""
    if hr is None:
        return None
    base = log[(log.person == person) & (log.kss <= 4)]
    base_hr = base.hr.dropna().to_numpy(float)
    if len(base_hr) < MIN_BASELINES:
        st.caption(f"Save {MIN_BASELINES - len(base_hr)} more check(s) on days you feel fine so the app learns "
                   "your normal heart rate.")
        return None
    hr_hi = base_hr.mean() + max(1.5 * base_hr.std(ddof=1), 5.0)
    msgs, load = [], hr > hr_hi
    msgs.append(f"Heart rate {hr:.0f} - your normal is about {base_hr.mean():.0f}"
                + (" (higher than usual)." if load else "."))
    base_rm = base.rmssd.dropna().to_numpy(float) if "rmssd" in base else np.array([])
    if rmssd is not None and len(base_rm) >= MIN_BASELINES:
        lg = np.log(base_rm)
        low = np.exp(lg.mean() - max(1.5 * lg.std(ddof=1), -np.log(1 - RETEST_RMSSD_DROP / 2)))
        if rmssd < low:
            load = True
            msgs.append("Heart-rate variability is below your normal range.")
    if load:
        st.info(" ".join(msgs) + " Your body is working harder than usual - could be stress, coffee, illness, heat "
                "or tiredness. Take a breather and check how you feel.")
    else:
        st.caption(" ".join(msgs))
    return load


def card(level, why, extra=()):
    icon, title, color, bg, todo = LEVELS[level]
    items = "".join(f"<li>{t}</li>" for t in list(todo) + list(extra))
    items = items.replace("**", "<b>", 1)
    while "**" in items:  # simple bold markup
        items = items.replace("**", "</b>", 1).replace("**", "<b>", 1)
    st.markdown(f"""<div class="card" style="background:{bg}; border:2px solid {color}">
<h2><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{color};margin-right:10px;vertical-align:middle"></span>{title}</h2><p class="why">{why}</p><ul>{items}</ul></div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ page
st.title("BreathBreak")
st.caption("Put the band on at the start of your shift. Your data stays on this computer.")
band_good_pct = None

log = load_log()
person = st.session_state.get("person", "me")
now = datetime.now()

# band check (start of shift + whole shift) -------------------------
STATUS_COLOR = {"good": "#22c55e", "loose": "#eab308", "too_tight": "#f97316", "no_contact": "#ef4444", "off": "#6b7280"}
STATUS_NAME = {"good": "good signal", "loose": "loose / moving", "too_tight": "too tight / light",
               "no_contact": "no contact", "off": "band taken off"}


def load_ppg(p):
    df = pd.read_csv(p, index_col=False)
    if "TIA1-3_Time" not in df:
        return None, None
    t, v = df["TIA1-3_Time"].to_numpy(float), df["TIA1-3_Value"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    return v, float(round(1.0 / np.median(np.diff(t))))


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def fix_box(status):
    title, why, steps = W.FIX[status]
    color = STATUS_COLOR[status]
    bg = "#dcfce7" if status == "good" else "#fee2e2" if status == "no_contact" else "#fef9c3"
    items = "".join(f"<li>{s}</li>" for s in steps)
    lst = f"<ol>{items}</ol>" if steps else ""
    st.markdown(f"""<div class="card" style="background:{bg}; border:2px solid {color}">
<h2 style="font-size:1.5rem"><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{color};margin-right:10px;vertical-align:middle"></span>{title}</h2>
<p class="why">{why}</p>{lst}</div>""", unsafe_allow_html=True)


def timeline(wins, summ, total_s):
    segs = []
    for w in wins:
        s = "off" if summ["off_at_s"] is not None and w.start_s >= summ["off_at_s"] else w.status
        segs.append((w.start_s, min(w.start_s + W.STEP_S, total_s), s))
    bar = "".join(f'<div title="{mmss(a)}-{mmss(b)}: {STATUS_NAME[s]}" style="flex:{b - a};background:{STATUS_COLOR[s]}"></div>'
                  for a, b, s in segs)
    legend = "".join(f'<span style="margin-right:14px"><span style="display:inline-block;width:10px;height:10px;'
                     f'border-radius:2px;background:{STATUS_COLOR[k]};margin-right:5px"></span>{STATUS_NAME[k]}</span>'
                     for k in STATUS_NAME)
    st.html(f"""<div style="font:14px system-ui,Segoe UI,sans-serif">
<div style="display:flex;gap:2px;height:26px;border-radius:6px;overflow:hidden">{bar}</div>
<div style="display:flex;justify-content:space-between;color:#6b7280;font-size:12px;margin-top:3px">
<span>0:00</span><span>{mmss(total_s)}</span></div>
<div style="margin-top:8px;color:#374151;font-size:13px">{legend}</div></div>""")


def band_plot(x, fs, summ):
    step = max(1, int(fs // 10))
    t = np.arange(0, len(x), step) / fs
    fig = go.Figure(go.Scatter(x=t, y=x[::step] * 1000, mode="lines", line=dict(width=1, color="#0d9488"),
                               hoverinfo="skip"))
    for p in summ["problems"]:
        fig.add_vrect(x0=p["from_s"], x1=p["to_s"], fillcolor=STATUS_COLOR[p["status"]], opacity=0.15, line_width=0)
    if summ["off_at_s"] is not None:
        fig.add_vrect(x0=summ["off_at_s"], x1=len(x) / fs, fillcolor="#6b7280", opacity=0.15, line_width=0)
    fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(title_text="seconds")
    return fig


st.subheader("1. Band check")
src = st.radio("Signal from", ["Simulated shift (demo)", "Example recording", "My file"], horizontal=True)
hrv = sig = path = None
heart_hint = None
hr_now = None
x = fs = None
if src == "My file":
    up = st.file_uploader("Upload the .csv from the band", type="csv")
    if up:
        REC_DIR.mkdir(parents=True, exist_ok=True)
        saved = st.session_state.get("saved_upload")
        if not saved or saved[0] != up.file_id:
            p = REC_DIR / f"{person}_{now:%Y%m%d_%H%M%S}_{Path(up.name).name}"
            p.write_bytes(up.getvalue())
            st.session_state.saved_upload = (up.file_id, p)
        path = st.session_state.saved_upload[1]
elif src == "Example recording":
    demo = sorted(DEMO_DIR.glob("*/*.csv")) if DEMO_DIR.exists() else []
    if demo:
        path = st.selectbox("Example", demo, format_func=lambda p: f"{p.parent.name} / {p.stem.replace('_', ' ')}")
    else:
        st.info("No example files on this computer.")
else:
    x, fs, _ = W.make_demo_shift()
    st.caption("Made-up 6-minute shift to show what the check does. Not real data.")

if path:
    x, fs = load_ppg(path)

if x is None and path:
    st.error("This file has no optical (PPG) channel.")
elif x is not None:
    total_s = len(x) / fs
    fit = W.fit_check(x, fs)
    st.markdown("**Start of shift** (first 20 seconds after putting the band on)")
    fix_box(fit.status)

    wins = W.scan(x, fs)
    summ = W.summarize(wins, total_s)
    st.markdown("**During the shift**")
    timeline(wins, summ, total_s)
    if summ["off_at_s"] == 0.0:
        st.error("The band never touched the skin in this recording.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Good signal", f"{summ['good_pct']:.0f}%", help="Share of the time the band was worn")
        c2.metric("Worn for", mmss(summ["worn_s"]))
        c3.metric("Taken off at", mmss(summ["off_at_s"]) if summ["off_at_s"] is not None else "still on")
    for prob in summ["problems"]:
        title, _, steps = W.FIX[prob["status"]]
        st.warning(f"{mmss(prob['from_s'])} - {mmss(prob['to_s'])}: {title.lower()}. {steps[0]}")
    if not summ["problems"] and summ["good_pct"] >= 80:
        st.success("No signal problems while the band was worn.")
    bpm = [w.pulse_bpm for w in wins if w.status == "good" and w.pulse_bpm]
    if bpm:
        st.write(f"Pulse while the signal was good: **{np.median(bpm):.0f} beats per minute**")
    if st.toggle("Show the signal"):
        st.plotly_chart(band_plot(x, fs, summ), width="stretch", config={"displayModeBar": False})

    # heart rate: ECG if the file has a good one, else the band's pulse.
    # heart-rate variability: ECG only, and only from >= 2 min (shorter clips vary 10-30 % by chance)
    hr_now = float(np.median(bpm)) if bpm else None
    if path:
        sig = read_ti_export(str(path), CFG)
        if sig["ecg"]:
            hrv = hrv_from_ecg(*sig["ecg"])
            if hrv is not None:
                hr_now = hrv["hr"]
                if sig["duration_s"] < MIN_HRV_S:
                    hrv["rmssd"] = None
                    st.caption(f"Heart-rate variability needs {MIN_HRV_S // 60} minutes sitting still "
                               f"(this recording: {sig['duration_s']:.0f} s). Only heart rate is used.")
    heart_hint = heart_vs_normal(log, person, hr_now, hrv["rmssd"] if hrv else None)
    band_good_pct = summ["good_pct"]


st.subheader("2. How are you today?")
c1, c2 = st.columns(2)
wake_t = c1.time_input("When did you wake up?", value=time(7, 0), step=900)
slept = c2.number_input("Hours of sleep last night", 0.0, 14.0, 7.0, 0.5)
c3, c4 = st.columns(2)
shift_t = c3.time_input("Shift started at", value=time(8, 0), step=900)
break_t = c4.time_input("Last break ended at", value=time(8, 0), step=900,
                        help="Use the shift start if you haven't had a break yet.")
feel = st.radio("How do you feel right now?", list(FEEL), index=1, horizontal=True)
kss = FEEL[feel]
st.caption("Anything else today?")
t1, t2, t3, t4 = st.columns(4)
unwell = t1.checkbox("Feeling unwell")
coffee = t2.checkbox("Coffee in last 3 h")
hot = t3.checkbox("Hot workplace")
no_water = t4.checkbox("No water for 2+ h")


def past(t):
    d = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    return d if d <= now else d - timedelta(days=1)


wake = past(wake_t)
on_shift_h = (now - past(shift_t)).total_seconds() / 3600
since_break_h = min((now - past(break_t)).total_seconds() / 3600, on_shift_h)
bundle = schedule_model()
feats = schedule_features(wake, now, slept)
risk = predict_fatigue(bundle, feats)
thr = bundle["alert_threshold"]

FEEL_RISK = {2: 0.10, 4: 0.25, 6: 0.50, 7: 0.70, 9: 0.90}   # same cut-offs as the schedule model
combined = max(risk, FEEL_RISK[kss])
level = 2 if combined >= thr else 1 if combined >= thr - 0.15 else 0
lvl_feel = 2 if kss >= 7 else 1 if kss >= 6 else 0
alertness = round(100 * (1 - combined))
awake = feats["hours_awake"]
reasons = [f"You've been awake {awake:.0f} hours and slept {slept:g}."]
if lvl_feel >= 1:
    reasons.append("You said you're " + {6: "a bit tired.", 7: "sleepy.", 9: "fighting sleep."}[kss])

extra = []


def at_least_yellow(why):
    global level, alertness
    if level == 0:
        level, alertness = 1, min(alertness, 55)
    reasons.append(why)


# rules, not learned weights: no dataset has these factors yet. They are logged so a model can learn them later.
if since_break_h >= BREAK_DUE_H:
    at_least_yellow(f"You've worked {since_break_h:.1f} hours without a break.")
if unwell:
    at_least_yellow("You're feeling unwell.")
    extra.append("If you feel worse, tell your supervisor and stop - being sick makes mistakes more likely.")
if heart_hint:
    at_least_yellow("Your heart is working harder than your normal.")
if on_shift_h >= LONG_SHIFT_H:
    reasons.append(f"You're {on_shift_h:.0f} hours into your shift.")
if coffee and level < 2:
    extra.append("You had coffee recently - it can hide tiredness for a few hours, so you may be more tired than this says.")
if hot or no_water:
    extra.append("Drink a glass of water now" + (" and keep drinking regularly in the heat." if hot else "."))

# rings -------------------------------------------------------------
def ring(frac, big, unit, label, color):
    r, c = 52, 2 * np.pi * 52
    frac = float(min(max(frac, 0), 1))
    return f"""<div class="ring"><svg viewBox="0 0 128 128" width="128" height="128">
<circle cx="64" cy="64" r="{r}" fill="none" stroke="#2b3440" stroke-width="10"/>
<circle cx="64" cy="64" r="{r}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
 stroke-dasharray="{c * frac:.1f} {c:.1f}" transform="rotate(-90 64 64)"/>
<text x="64" y="72" text-anchor="middle" font-size="30" font-weight="700" fill="#fff"
 font-family="system-ui,Segoe UI,sans-serif">{big}<tspan font-size="16" dx="1">{unit}</tspan></text></svg>
<div class="lbl">{label}</div></div>"""


sleep_pct = min(slept / 8.0, 1.0)
level_color = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}[level]
st.html(f"""<style>
.rings {{display:flex; justify-content:space-around; flex-wrap:wrap; gap:8px; background:#151b23;
        border-radius:18px; padding:20px 8px 14px;}}
.ring {{text-align:center; min-width:120px;}}
.ring .lbl {{color:#e5e7eb; font:600 13px/1.2 system-ui,Segoe UI,sans-serif; letter-spacing:.12em; margin-top:4px;}}
</style><div class="rings">
{ring(sleep_pct, round(sleep_pct * 100), "%", "SLEEP", "#93b4d6")}
{ring(alertness / 100, alertness, "%", "ALERTNESS", level_color)}
{ring(awake / 16, f"{awake:.1f}", "h", "AWAKE", "#0ea5e9")}
</div>""")
st.caption("Sleep = hours slept vs 8 h · Alertness = from your sleep, hours awake, time of day and how you feel · "
           "Awake = hours since you woke (ring full at 16 h)")

# answer ------------------------------------------------------------
st.subheader("3. Your result")
card(level, " ".join(reasons), extra)

if st.button("Save today's check", type="primary", width="stretch"):
    row = {"file": (str(Path(path).relative_to(ROOT)) if path and Path(path).is_relative_to(ROOT) else str(path or "")),
           "person": person, "recorded_at": now.isoformat(timespec="minutes"), "wake": wake_t.strftime("%H:%M"),
           "slept": slept, "kss": kss, "risk": round(risk, 3),
           "hr": None if hr_now is None else round(hr_now, 1),
           "rmssd": round(hrv["rmssd"], 1) if hrv and hrv["rmssd"] is not None else None,
           "shift_start": shift_t.strftime("%H:%M"), "hours_on_shift": round(on_shift_h, 2),
           "hours_since_break": round(since_break_h, 2), "unwell": unwell, "coffee_3h": coffee,
           "hot": hot, "no_water_2h": no_water,
           "ecg_ok": bool(sig and sig["ecg"]), "ppg_ok": bool(sig and sig["ppg"]),
           "band_good_pct": None if band_good_pct is None else round(band_good_pct),
           "condition": "fatigue" if kss >= 7 else ("normal" if kss <= 4 else "")}
    APP_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)
    st.success("Saved. Thanks — each check helps the app learn your normal.")

with st.expander("Settings & history"):
    st.session_state.person = st.text_input("Your name", value=person).strip() or "me"
    mine = log[log.person == person]
    if len(mine):
        show = mine.assign(feeling=mine.kss.map({v: k for k, v in FEEL.items()}))
        st.dataframe(show.reindex(columns=["recorded_at", "slept", "feeling", "hr"]).sort_values("recorded_at", ascending=False)
                     .rename(columns={"recorded_at": "when", "slept": "sleep (h)", "hr": "heart rate"}),
                     hide_index=True, width="stretch")
    else:
        st.write("No saved checks yet.")
    st.caption("How it works: the result uses your sleep, hours awake and time of day (tested on a public "
               "sleep-study dataset) and how you say you feel. Time since your last break, feeling unwell and a heart "
               "rate above your own normal can raise it to yellow. Heart data never makes it red.")
