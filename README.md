# EnsembleAI

**Find out who in your ensemble is out of tune, and when.**

When a band or orchestra sounds off, the director has to stop everyone and hunt for the problem by ear: which section, which passage, sharp or flat? EnsembleAI answers that from a single recording. Upload a short clip, list the sections, and it isolates each one, measures every held note against the rest of the group, and tells the director what to fix first.

Built for the **Meta Global AI Developer Hackathon** (Indie track, *Build for Your World*).

**Demo video:** <!-- YouTube/Loom link -->  
**Live demo (Hugging Face Space):** <!-- Space link --> (replays real results from the full pipeline, so it runs without a GPU)

<!-- Add a screenshot of the ensemble report here: ![EnsembleAI report](screenshot.png) -->

## What it does

- **Ensemble report.** Isolates each section with **Meta SAM Audio**, then shows a timeline with a lane per section: green is in tune, red is more than 15 cents off the group, labeled sharp (+) or flat (−).
- **Verdict at a glance.** For example: "Clarinet is running sharp: about +22 cents on 9 of 12 held notes."
- **Hear the problem.** Plays just the worst passage from the isolated section.
- **Rehearsal notes from Muse Spark.** Turns the measurements into practical notes, such as which passage to fix first and what to try.
- **Ask the rehearsal assistant.** Ask in plain English ("Is anyone sharp in the last five seconds?" or "Is the euphonium sharp on the third note?"). **Muse Spark plans** which sections and time window to check, **runs EnsembleAI's measurement tools**, and **answers only from the measured results**.
- **Solo mode.** List one section and it measures the recording directly, with no separation step.
- **Honest when it can't tell.** Sections that can't be isolated are marked "couldn't isolate clearly"; if nothing can be judged, it says "No verdict" instead of guessing. Typos and empty inputs get a friendly explanation instead of an error.

## How it works

```
recording ──► Meta SAM Audio (text prompt per section) ──► isolated section + "everyone else"
                                                              │
              pitch tracking (librosa pYIN + YIN) ◄──────────┘
                     │
                     ▼
     cents off, measured against the GROUP's tuning (not a fixed A440)
                     │
                     ▼
  timeline + verdict + worst passage ──► Muse Spark: rehearsal notes / answers questions
```

**Design decisions we made during the hackathon:**

1. **Measure against the group, not A440.** Ensembles often tune slightly above or below A440 (our test recording sat about 17 cents sharp). A director cares whether a section matches the group, so every note is compared to the ensemble's own tuning.
2. **Find the odd one out.** A single out-of-tune section can drag a naive "group average" toward itself and make in-tune sections look wrong. For each section we compare its tuning to everyone else's; the section that disagrees most is set aside, and everyone is measured against the remaining group.
3. **Don't let a muddy track vote.** Sections that couldn't be isolated cleanly are shown as "couldn't isolate clearly" and excluded from choosing the reference, instead of being guessed at.
4. **Do the math in code; let the language model write words.** Early on, Muse Spark "corrected" a measurement it was given and reported a 22-cent-sharp note as in tune. Now it only receives final verdicts ("OUT OF TUNE, 22 cents SHARP") and never raw numbers it could misapply.
5. **Credit leaked sound to the right section.** Loud instruments leak into other sections' isolated tracks. When two sections show the same out-of-tune note at the same moment, it's credited only to the track where it's loudest, and a single stray note never counts as a section "running sharp."
6. **Keep working if the language model doesn't.** If the Muse Spark call fails, the report still shows every measurement.
7. **Fit on a free GPU.** SAM Audio's optional re-ranking and span-prediction models pushed free Colab over its memory limit. We load only what single-candidate separation needs and load weights directly onto the GPU.

## Results

| Test | Injected | Measured | Result |
|---|---|---|---|
| Mozart Clarinet Quintet, original | none | all sections within 15 cents | In tune ✓ |
| Same recording, clarinet isolated, pitch-shifted, and remixed | clarinet +25 cents | clarinet +18 to +30 (median about +22); cello in tune | Clarinet flagged ✓ |
| Mozart String Quartet, cello detuned | cello +25 cents | cello +18 to +25 | Cello flagged ✓ |
| Real duet recording (euphonium + bassoon), third note known to be sharp | (real performance) | third note +33 cents, the worst note in the opening | Found ✓ |
| Whole quintet shifted together | everyone +30 cents | all sections in tune | Correct: the group is together ✓ |
| Two sections off in opposite directions | clarinet +25, cello −25 | clarinet +23 flagged; cello couldn't be isolated | Partial: loudest culprit found |
| Very large detune | clarinet +60 cents | clarinet flagged as about −39; some leakage into the violin track | Partial: see limitations |

**Leakage handling:** in the two-sections test, crediting identical out-of-tune notes to the loudest track removed the violin false alarm (10 of 12 notes falsely flagged before, 1 stray note after). In-tune notes are never touched, so recordings with no problems are unaffected.

We also stress-tested bad input: instruments not in the recording, non-music audio, typos, empty boxes, off-topic questions to the assistant ("What's the weather?" gets "I can't tell the weather from these results"), very short clips, and mp3/m4a/wav files. See [`TEST_LOG.md`](TEST_LOG.md) for every test, what went wrong, and how it was fixed.

## Built during the hackathon vs. third-party

**We built (Oct 26 – Nov 4, 2026):** the tuning analysis pipeline, group-relative measurement, odd-one-out reference selection, unclear-section handling, leakage handling, the ensemble timeline and verdict, worst-passage playback, solo mode, input validation, the Muse Spark planning-and-answer loop, the rehearsal-notes prompts, the detune test harness, the Gradio app, and the hosted replay demo.

**Third-party models and libraries:** Meta SAM Audio (`facebook/sam-audio-small`), Muse Spark via the Meta Model API, librosa, NumPy, matplotlib, Gradio, PyTorch.

## Repo contents

| File | What it is |
|---|---|
| `ensembleai_core.py` | Models and measurement: SAM Audio loading and separation, pitch analysis, reference selection, Muse Spark calls |
| `ensembleai_app.py` | The Gradio app (ensemble report + rehearsal assistant) |
| `EnsembleAI.ipynb` | Colab notebook that runs everything in 3 cells |
| `export_demo_results.py` | Saves real results for the hosted demo |
| `space/` | The hosted demo: a static page that replays those results without a GPU (`demo_results/` lives on the Space) |
| `cq.wav`, `cq_detuned.wav` | Demo clips (CC BY-SA, see credits) |
| `TEST_LOG.md` | Every test we ran and how each problem was resolved |

## Run it

Requires a GPU (a Colab T4 works) and Python 3.11+.

1. Request access to [`facebook/sam-audio-small`](https://huggingface.co/facebook/sam-audio-small) on Hugging Face and create a read token.
2. Get a Meta Model API key at [dev.meta.ai](https://dev.meta.ai).
3. In Colab, set the runtime to **T4 GPU** and add two secrets (key icon on the left): `HF_TOKEN` and `MODEL_API_KEY`.
4. Upload `ensembleai_core.py`, `ensembleai_app.py`, and the demo clips, then run three cells:

```python
# Cell 1: install (the session restarts itself afterward; that's expected)
!pip install -q git+https://github.com/facebookresearch/sam-audio.git openai
import os; os.kill(os.getpid(), 9)
```
```python
# Cell 2: load SAM Audio, the measurement tools, and Muse Spark
exec(open("ensembleai_core.py").read())
```
```python
# Cell 3: launch the app (prints a public link)
exec(open("ensembleai_app.py").read())
```

Never commit API keys or tokens.

## Limitations

- **Similar instruments are hard to separate.** Viola and violin in a string quartet blur together; contrasting instruments (clarinet vs. strings) separate well. The app says when a section couldn't be isolated instead of guessing.
- **SAM Audio's vocabulary is limited.** It didn't recognize "euphonium" or match it to "tuba," so solo passages on rarer instruments should use solo mode.
- **Short clips.** We analyze the first 15 seconds because of free-GPU memory limits.
- **Held notes only.** Very fast passages are less reliable.
- **More than half a step off.** A note 60 cents sharp reads as about 40 cents flat of the next note up. Every tuner has this limit, and a player that far off sounds like a wrong note anyway.
- **Several sections off at once.** The odd-one-out reference reliably catches the section that disagrees most; a second culprit can be missed, especially if it doesn't separate cleanly.

## What's next

- **Score-aware analysis:** upload sheet music to tell "wrong note" apart from "right note, out of tune."
- **Live rehearsal mode:** feedback while the ensemble plays.
- **Individual players** within a section.
- **Chord-aware tuning,** since ensembles don't always tune to equal temperament.

## Credits and licenses

**Code:** MIT (see `LICENSE`).

**Demo audio:** "Wolfgang Amadeus Mozart - Clarinet Quintet - 2. Larghetto" by <!-- Author from the file page -->, via [Wikimedia Commons](<!-- file page link -->), licensed [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/). Modified: trimmed to 15 seconds; for the detuned demo, the clarinet was isolated and pitch-shifted +25 cents. **Our derived audio clips are released under CC BY-SA 2.0.**

<!-- Add the same credit for the Mozart String Quartet K421 recording if it appears in the repo or video. -->

**Duet recording:** performed by <!-- name(s) -->, used with permission.

## Team

<!-- Names and GitHub handles -->
