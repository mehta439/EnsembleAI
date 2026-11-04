# EnsembleAI core: models + measurement tools.
# Run once per session, AFTER installing sam-audio and restarting:  exec(open("ensembleai_core.py").read())
import os
import numpy as np, librosa, soundfile as sf, matplotlib.pyplot as plt
import torch, torchaudio
from openai import OpenAI

def _secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

# ---------- Hugging Face login (SAM Audio weights are gated) ----------
from huggingface_hub import login
if _secret("HF_TOKEN"):
    login(_secret("HF_TOKEN"))
else:
    print("No HF_TOKEN secret found. If loading fails, run: from huggingface_hub import notebook_login; notebook_login()")

# ---------- Meta SAM Audio (loaded once; reloading on top of itself runs out of RAM) ----------
from sam_audio import SAMAudio, SAMAudioProcessor

def _patch(cls):   # newer huggingface_hub stopped passing these two arguments
    if hasattr(cls, "_from_pretrained") and not getattr(cls, "_ensembleai_patched", False):
        orig = cls._from_pretrained.__func__
        def fixed(c, *a, proxies=None, resume_download=None, **kw):
            return orig(c, *a, proxies=proxies, resume_download=resume_download, **kw)
        cls._from_pretrained = classmethod(fixed)
        cls._ensembleai_patched = True
_patch(SAMAudio); _patch(SAMAudioProcessor)

MODEL_NAME = "facebook/sam-audio-small"
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then reinstall, "
                       "restart the session, and re-upload your files (changing runtime wipes them).")
if globals().get("model") is None:
    # Skip the re-ranking and span-prediction models: not needed for 1 candidate, and they crash free Colab
    model = SAMAudio.from_pretrained(MODEL_NAME, visual_ranker=None, text_ranker=None,
                                     span_predictor=None, map_location="cuda").eval().cuda()
    processor = SAMAudioProcessor.from_pretrained(MODEL_NAME)
    print("SAM Audio loaded.")
else:
    print("SAM Audio already loaded; skipping.")

def separate(path, instrument, out_target, out_rest):
    torch.manual_seed(0)   # same result every run
    batch = processor(audios=[path], descriptions=[instrument]).to("cuda")
    with torch.inference_mode():
        r = model.separate(batch, predict_spans=False, reranking_candidates=1)
    sr = processor.audio_sampling_rate
    torchaudio.save(out_target, r.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(out_rest, r.residual[0].unsqueeze(0).cpu(), sr)

# ---------- Tuning measurement ----------
RANGES = {"cello": ("C2", "A5"), "viola": ("C3", "E6"), "violin": ("G3", "A7"), "double bass": ("E1", "G4"),
          "clarinet": ("D3", "C7"), "flute": ("C4", "C7"), "oboe": ("A#3", "A6"), "bassoon": ("A#1", "E5"),
          "saxophone": ("C#3", "A5"), "trumpet": ("F#3", "D6"), "trombone": ("E2", "F5"),
          "french horn": ("B1", "F5"), "horn": ("B1", "F5"), "euphonium": ("A#1", "F5"),
          "baritone horn": ("A#1", "F5"), "tuba": ("D1", "F4"), "piano": ("A0", "C8")}

def group_tuning(path):
    """Overall tuning of a recording, in cents from A440 (works with many instruments at once)."""
    y, sr = librosa.load(path, sr=22050)
    return float(librosa.estimate_tuning(y=y, sr=sr) * 100)

def analyze(path, ref, label, instrument="cello", show=True):
    """Held notes in one isolated track, each measured in cents against the reference tuning `ref`."""
    y, sr = librosa.load(path, sr=22050)
    lo, hi = (librosa.note_to_hz(n) for n in RANGES.get(instrument.lower(), ("C2", "C6")))
    _, voiced, prob = librosa.pyin(y, fmin=lo, fmax=hi, sr=sr)   # where notes are being played
    f0 = librosa.yin(y, fmin=lo, fmax=hi, sr=sr)                  # exact pitch, no rounding
    times = librosa.times_like(f0, sr=sr)
    midi = librosa.hz_to_midi(f0)
    cents = (midi - np.round(midi - ref / 100)) * 100 - ref
    note = np.where(voiced & (prob > 0.3), np.round(midi - ref / 100), np.nan)

    segs, start = [], None
    for i, n in enumerate(note):
        if start is not None and (np.isnan(n) or n != note[start]):
            segs.append((start, i)); start = None
        if start is None and not np.isnan(n):
            start = i
    if start is not None:
        segs.append((start, len(note)))

    rows = []
    if show:
        print(f"\n{label}  (reference tuned {ref:+.1f} cents from A440)")
    for s, e in segs:
        if times[e - 1] - times[s] < 0.15:
            continue
        c = float(np.median(cents[s:e]))
        rows.append({"start": round(float(times[s]), 2), "end": round(float(times[e - 1]), 2),
                     "note": librosa.midi_to_note(int(note[s])), "cents": round(c, 1)})
        if show:
            print(f"{times[s]:5.2f}-{times[e-1]:5.2f}s  {rows[-1]['note']:5}  {c:+5.1f} cents   "
                  f"{'OUT OF TUNE' if abs(c) > 15 else 'ok'}")
    plt.figure(figsize=(12, 3))
    plt.scatter(times, np.where(np.isnan(note), np.nan, cents), s=4)
    plt.axhspan(-15, 15, alpha=0.15, color="green")
    plt.ylim(-50, 50); plt.title(label); plt.xlabel("seconds"); plt.ylabel("cents off")
    analyze.last_fig = plt.gcf()
    plt.show() if show else plt.close()
    return rows

def pick_reference(paths):
    """paths = {section: (isolated_track, everyone_else_track)}. Returns (reference cents, explanation).
    Finds the section that disagrees most with everyone else and measures all sections against the others."""
    wrap = lambda c: ((c + 50) % 100) - 50
    own = {s: group_tuning(t) for s, (t, r) in paths.items()}
    rest = {s: group_tuning(r) for s, (t, r) in paths.items()}
    gap = {s: wrap(own[s] - rest[s]) for s in paths}
    odd = max(gap, key=lambda s: abs(gap[s]))
    if abs(gap[odd]) > 15:
        return rest[odd], f"{odd} disagrees most with the group, so every section is measured against the others"
    return float(np.median(list(rest.values()))), "no section stands out, so everyone is measured against the whole group"

# ---------- Muse Spark (Meta Model API) ----------
client = OpenAI(api_key=_secret("MODEL_API_KEY"), base_url="https://api.meta.ai/v1")
MUSE_MODEL = "muse-spark-1.3"

def muse(prompt):
    """One Muse Spark call. Returns None instead of crashing if the API fails."""
    try:
        resp = client.chat.completions.create(model=MUSE_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except Exception as e:
        print("Muse Spark call failed:", e)
        return None

def rehearsal_notes_ensemble(results):
    lines = []
    for sec, rows in results.items():
        if not rows:
            lines.append(f"{sec}: couldn't be isolated clearly, no verdict")
            continue
        for r in rows:
            if abs(r["cents"]) > 15:
                d = "SHARP" if r["cents"] > 0 else "FLAT"
                lines.append(f"{sec}, {r['start']}-{r['end']}s, note {r['note']}: OUT OF TUNE, {abs(r['cents']):.0f} cents {d}")
        if not any(abs(r["cents"]) > 15 for r in rows):
            lines.append(f"{sec}: in tune throughout")
    out = muse("You help an ensemble director run rehearsal. Below are tuning results for each section, "
               "already compared to the group's tuning. Each verdict is FINAL; do not recalculate. Only "
               "mention sections listed below.\n\n" + "\n".join(lines) +
               "\n\nWrite 2-4 short, practical rehearsal notes: which section and passage to fix first, "
               "sharp or flat, and a concrete fix. Mention sections that are in tune in one line. No preamble.")
    return out or "*Rehearsal notes are unavailable right now (Muse Spark didn't respond). The measurements above are complete.*"

print("EnsembleAI core ready.")
