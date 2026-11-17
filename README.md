# BreathBreak: fatigue models

Two models, as on slide 4:

| | Model A | Model B (transfer learning) |
|---|---|---|
| Input | 60-s ECG (or PPG) window | 60-s wrist PPG window |
| Features | Time-domain HRV: mean HR, SDNN, RMSSD, pNN50, CVNN | Embedding from a **frozen** pretrained PPG model (PaPaGei / Pulse-PPG) |
| Classifier | Logistic regression | Logistic regression on the embedding (linear probe) |
| Train on | DROZY (+ your AFE4950 data) | MEFAR (+ your AFE4950 data) |
| Personal baseline | log(feature / own 2-min baseline) | embedding − own baseline embedding |

Both are evaluated **leave-one-subject-out**: every person is scored by a model that never saw them.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Test everything on fake data first (2 minutes)

```bash
python scripts/make_synthetic_data.py
python scripts/train_model_a.py --own data_synth/own --wesad data_synth/WESAD
python scripts/train_model_b.py --mefar data_synth/MEFAR_raw --mefar-labels data_synth/mefar_labels.csv --embedder prv --wesad data_synth/WESAD
python scripts/realtime_demo.py --model models/model_a.joblib --csv data_synth/own/p3_rested.csv data_synth/own/p3_late.csv --fs 250
```
Synthetic scores will be ~100%. That only proves the code runs.

## 3. Get the real data

| Dataset | Where | Put it in |
|---|---|---|
| DROZY | https://orbi.uliege.be/handle/2268/191620 (sign licence) | `data/DROZY/` with `psg/<subj>-<sess>.edf` + `KSS.txt` |
| MEFAR | https://data.mendeley.com/datasets/z3g26tphnv | `data/MEFAR_raw/<id>/{morning,evening}/BVP.csv` |
| WESAD | https://archive.ics.uci.edu/dataset/465/wesad | `data/WESAD/S2/S2.pkl` ... |

**DROZY layout differs from mine?** Edit `edf_pattern` in `breathbreak/data/drozy.py`, or make `data/drozy_labels.csv` (`subject,session,kss`) and pass `--drozy-labels`. The loader picks the first channel whose name contains "ECG".

**MEFAR labels:** open `general_info.xlsx`, copy each participant's Chalder Fatigue Scale score for the morning and evening sheets into `data/mefar_labels.csv`:
```
subject,session,cfs
01,morning,9
01,evening,15
```
(`subject` must match the folder name exactly, e.g. `01`.) CFS ≥ 12 → fatigued.

## 4. Train on real data

```bash
# Model A
python scripts/train_model_a.py --drozy data/DROZY --wesad data/WESAD

# Model B, start with the no-weights baseline, then the foundation model
python scripts/train_model_b.py --mefar data/MEFAR_raw --mefar-labels data/mefar_labels.csv --embedder prv --wesad data/WESAD
python scripts/train_model_b.py --mefar data/MEFAR_raw --mefar-labels data/mefar_labels.csv --embedder papagei --wesad data/WESAD
```

### PaPaGei setup (for `--embedder papagei`)
```bash
git clone https://github.com/nokia-bell-labs/papagei-foundation-model third_party/papagei-foundation-model
pip install -r third_party/papagei-foundation-model/requirements.txt
# download papagei_s.pt from Zenodo record 13983110 → weights/papagei_s.pt
```
⚠ Check `PaPaGeiEmbedder.MODEL_CONFIG` in `breathbreak/embedders.py` against the README quick-start. This path was **not** run in testing (no weights available).

### Pulse-PPG (`--embedder pulseppg`)
The best fit for wrist PPG, but its README doesn't document the embedding API. Fill in `_load()` and `_forward()` in `PulsePPGEmbedder` from the repo's Colab demo (https://github.com/maxxu05/pulseppg).

## 5. Read the results honestly

Each run prints three things:

1. **Model, leave-one-subject-out:** balanced accuracy, F1, AUC, and per-person accuracy (median, min, max). Report the **per-person range**, not just the average. With 14 people, ±20 points of noise is normal.
2. **Control, session order only:** a model that only knows *which session* a window came from. In DROZY the alert session is always first, in the morning. **If this control scores close to your model, your model is learning time-of-day, not fatigue.** Say so on the slide.
3. **Stress control (WESAD):** nobody there is sleep-deprived. P(fatigued) should **not** rise from baseline to stress. If it does, the model confuses stress with fatigue.

Per-window predictions: `reports/model_a_windows.csv`. Summaries: `reports/*.json`.

## 6. Your own AFE4950 recordings

Export each recording as CSV (`time_s,ecg,ppg`) into `data/own/`, plus `data/own/sessions.csv`:
```
subject,session,file,fs,kss,is_baseline
nhat,rested,nhat_rested.csv,500,3,1
nhat,late,nhat_late.csv,500,8,0
```
Then add `--own data/own` to either training script. At each recording, ask the person for a **KSS** score (1–9 sleepiness).

## 7. Run on the device (SK-AM62)

`breathbreak/realtime.py` → `RealtimeScorer`:
```python
from breathbreak.realtime import RealtimeScorer
scorer = RealtimeScorer.from_file("models/model_a.joblib", input_fs=500)
while True:
    chunk = read_one_second_from_afe4950()       # your driver code
    for ev in scorer.push(chunk):
        if ev["type"] == "ALERT": buzz_and_show_breathing_prompt()
```
- First 2 min: baseline (tell the worker to sit still, or hold the finger pad for ECG).
- Then one score every 60 s; noisy windows are **skipped**, not guessed.
- Alert after 2 consecutive smoothed scores ≥ 0.6 (tune in `config.py`).

The realtime scorer supports Model A bundles (HRV features). For Model B on the device, call the same embedder on each 60-s PPG window and `bundle["model"].predict_proba()`.

## 8. Settings worth knowing (`breathbreak/config.py`)

- `kss_fatigue_threshold = 7`: KSS ≥ 7 counts as fatigued.
- `model_a_features`: time-domain only. LF/HF needs about 180 s and is unreliable at 60 s (Shaffer et al. 2020).
- `ppg_fs = 64`: set the **AFE4950 PPG to ≥ 100 Hz** when recording. Low rates (25 Hz on Galaxy Watch) inflate RMSSD.

## Licences
DROZY and WESAD: non-commercial research only. MEFAR: CC BY 4.0. Fine for the hackathon; a commercial product must retrain on MEFAR plus your own data.

## Files
```
breathbreak/
  config.py        all settings
  signals.py       filtering, resampling, R-peak / PPG-peak detection, IBI cleaning
  features.py      60-s windows, HRV features, baseline normalization
  dataset.py       recordings -> feature table (A) / raw windows (B)
  embedders.py     prv | papagei | pulseppg (frozen, transfer learning)
  models.py        Model A / Model B pipelines, save/load
  evaluate.py      leave-one-subject-out, session-order control, stress check
  realtime.py      device-side scorer with baseline + alerting
  data/            loaders: drozy, mefar, wesad, own
scripts/
  train_model_a.py  train_model_b.py  realtime_demo.py  make_synthetic_data.py
```
