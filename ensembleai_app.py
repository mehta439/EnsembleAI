import gradio as gr, json, re, hashlib, os, inspect
import matplotlib.pyplot as plt

# ---------- shared helpers ----------
_sep_cache = {}
def separate_cached(path, sec):
    key = (hashlib.md5(open(path, "rb").read()).hexdigest(), sec)
    if key not in _sep_cache:
        t, r = f"{key[0][:8]}_{sec}_t.wav", f"{key[0][:8]}_{sec}_r.wav"
        separate(path, sec, t, r)
        _sep_cache[key] = (t, r)
    return _sep_cache[key]

def measure_ensemble(audio_path, sections):
    y, sr = librosa.load(audio_path, sr=None, duration=15)
    sf.write("ui_in.wav", y, sr)
    if len(sections) == 1:   # solo mode: no separation, measure against the whole recording's tuning
        s = sections[0]
        yfull, srf = librosa.load(audio_path, sr=22050)
        ref = float(librosa.estimate_tuning(y=yfull, sr=srf) * 100)
        return ({s: analyze("ui_in.wav", ref, s, s, show=False)}, {s: ("ui_in.wav", "ui_in.wav")},
                f"one section listed, so the recording is measured directly (no separation) against "
                f"the whole recording's tuning ({ref:+.0f} cents from A440)")
    paths = {s: separate_cached("ui_in.wav", s) for s in sections}
    ref, why = pick_reference(paths)
    measured = {s: analyze(paths[s][0], ref, s, s, show=False) for s in sections}
    unclear = [s for s in sections if not measured[s]]
    if unclear and len(sections) - len(unclear) >= 2:   # don't let a muddy track pick the reference
        ref, why = pick_reference({s: p for s, p in paths.items() if s not in unclear})
        measured = {s: analyze(paths[s][0], ref, s, s, show=False) for s in sections}
    measured, n_dropped = drop_leakage(measured, paths)
    if n_dropped:
        why += (f"; {n_dropped} note{'s' if n_dropped != 1 else ''} heard identically in two tracks "
                "were credited to the louder one (sound leaking between tracks)")
    return measured, paths, why

def drop_leakage(measured, paths):
    """The same out-of-tune sound often leaks into several isolated tracks. If two sections show the same
    OUT-OF-TUNE note at the same time, off in the same direction, credit it only to the track where it's
    loudest. In-tune notes are never touched, so a recording with no problems is unaffected."""
    audio = {s: librosa.load(paths[s][0], sr=22050)[0] for s in measured}
    def loud(sec, a, b):
        y = audio[sec][int(a * 22050):int(b * 22050) + 1]
        return float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    drop = {s: set() for s in measured}
    for s, rows in measured.items():
        for j, r in enumerate(rows):
            for o, rows2 in measured.items():
                if o == s:
                    continue
                for r2 in rows2:
                    overlap = min(r["end"], r2["end"]) - max(r["start"], r2["start"])
                    shorter = min(r["end"] - r["start"], r2["end"] - r2["start"])
                    both_off_same_way = (abs(r["cents"]) > 15 and abs(r2["cents"]) > 15
                                         and (r["cents"] > 0) == (r2["cents"] > 0))
                    if (r["note"] == r2["note"] and both_off_same_way and abs(r["cents"] - r2["cents"]) < 12
                            and shorter > 0
                            and overlap > 0.5 * shorter
                            and loud(o, r2["start"], r2["end"]) > loud(s, r["start"], r["end"])):
                        drop[s].add(j)
    kept = {s: [r for j, r in enumerate(rows) if j not in drop[s]] for s, rows in measured.items()}
    return kept, sum(len(d) for d in drop.values())

class InputProblem(Exception):
    pass

ALIASES = {"violins": "violin", "cellos": "cello", "violas": "viola", "clarinets": "clarinet",
           "flutes": "flute", "oboes": "oboe", "bassoons": "bassoon", "trumpets": "trumpet",
           "trombones": "trombone", "horns": "horn", "tubas": "tuba", "sax": "saxophone",
           "saxophones": "saxophone", "bass": "double bass", "upright bass": "double bass",
           "contrabass": "double bass", "baritone": "baritone horn", "euph": "euphonium"}

def parse_sections(text):
    out = []
    for s in (text or "").split(","):
        s = " ".join(s.strip().lower().split())
        if not s:
            continue
        s = ALIASES.get(s, s)
        if s not in RANGES:
            raise InputProblem(f"“{s}” isn't an instrument EnsembleAI knows. Try one of: "
                               + ", ".join(sorted(RANGES)) + ".")
        if s not in out:                 # drop duplicates
            out.append(s)
    if not out:
        raise InputProblem("List at least one section, for example “clarinet, violin, cello”.")
    return out

def problem_card(msg):
    return f"<div class='verdict warn'><div class='v-head'>Can't analyze yet</div><div class='v-line'>{msg}</div></div>"

RED, GREEN, INK, MUTED = "#e5484d", "#30a46c", "#000000", "#3d444d"
def timeline(results, empty_msg="couldn't isolate clearly enough to judge"):
    fig, ax = plt.subplots(figsize=(12, max(2.4, 1.2 + 0.85 * len(results))), facecolor="white")
    ends = [r["end"] for rows in results.values() for r in rows]
    xmax = max(ends) + 0.3 if ends else 15
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(results) - 0.4)   # padding so labels never sit on the axis
    ax.set_facecolor("white")
    for i, (sec, rows) in enumerate(results.items()):
        if not rows:
            msg = empty_msg.get(sec, "") if isinstance(empty_msg, dict) else empty_msg
            ax.text(0.01, i, msg, transform=ax.get_yaxis_transform(),
                    va="center", color=MUTED, fontsize=11, style="italic")
        for r in rows:
            bad = abs(r["cents"]) > 15
            ax.barh(i, max(r["end"] - r["start"], 0.05), left=r["start"], height=0.62,
                    color=RED if bad else GREEN, alpha=0.9 if bad else 0.55, edgecolor="white", linewidth=1.5)
            if bad:
                mid, label = (r["start"] + r["end"]) / 2, f'{r["cents"]:+.0f}'
                if r["end"] - r["start"] > 0.028 * xmax:      # wide enough: number inside the bar
                    ax.text(mid, i, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                else:                                          # narrow bar: number just above it
                    ax.text(mid, i + 0.34, label, ha="center", va="bottom", color=RED, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([s.capitalize() for s in results], fontsize=13, fontweight="bold")
    ax.set_xlabel("seconds", color=INK, fontsize=11)
    ax.tick_params(colors=INK, labelcolor=INK, labelsize=11, length=0)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    ax.grid(axis="x", color="#eaeef2", linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("Red = more than 15 cents off the group (number = cents sharp + or flat −)",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig

def verdict_html(results, why):
    flagged, slips, unclear = [], 0, []
    for sec, rows in results.items():
        if not rows:
            unclear.append(sec); continue
        bad = [r for r in rows if abs(r["cents"]) > 15]
        if len(bad) >= 2 and len(bad) / len(rows) >= 0.3:   # one stray note isn't a pattern
            c = float(np.median([r["cents"] for r in bad]))
            flagged.append((sec, c, len(bad), len(rows)))
        else:
            slips += len(bad)
    if not flagged and len(unclear) == len(results):
        return ("<div class='verdict neutral'><div class='v-head'>No verdict</div>"
                "<div class='v-line'>No section could be isolated clearly enough to judge.</div>"
                "<div class='v-note'>Check that the instruments listed are actually in the recording, "
                "and that it's music with sustained notes.</div>"
                f"<div class='v-note'>How we judged: {why}</div></div>")
    if flagged:
        items = "".join(
            f"<div class='v-line'><b>{s.capitalize()}</b> is running <b>{'sharp' if c > 0 else 'flat'}</b>: "
            f"about <b>{c:+.0f} cents</b> on {n} of {m} held notes</div>" for s, c, n, m in flagged)
        head, cls = "Out of tune", "bad"
    else:
        items = ("<div class='v-line'>Every section we could measure is within 15 cents of the group" if unclear
                 else "<div class='v-line'>Every section is within 15 cents of the group")
        items += f" (a few brief slips: {slips})</div>" if slips else ".</div>"
        head, cls = "In tune", "good"
    extra = f"<div class='v-note'>Couldn't isolate clearly: {', '.join(unclear)}</div>" if unclear else ""
    return (f"<div class='verdict {cls}'><div class='v-head'>{head}</div>{items}{extra}"
            f"<div class='v-note'>How we judged: {why}</div></div>")

def ask_json(prompt):
    text = muse(prompt) or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}

# ---------- tab 1: ensemble report ----------
def run_report(audio_path, sections_text):
    try:
        if not audio_path:
            raise InputProblem("Upload a recording first, or click one of the demo recordings.")
        sections = parse_sections(sections_text)
    except InputProblem as e:
        return problem_card(str(e)), None, None, "", "", None
    results, paths, why = measure_ensemble(audio_path, sections)
    table, worst = [], None
    for sec, rows in results.items():
        for r in rows:
            bad = abs(r["cents"]) > 15
            table.append([sec, f"{r['start']}–{r['end']}s", r["note"], r["cents"], "OUT OF TUNE" if bad else "ok"])
            if bad and (worst is None or abs(r["cents"]) > abs(worst[1]["cents"])):
                worst = (sec, r)
    worst_audio, worst_md = None, "No out-of-tune passages to play."
    if worst:
        sec, r = worst
        ys, srs = librosa.load(paths[sec][0], sr=None)
        a, b = int(max(r["start"] - 0.5, 0) * srs), int((r["end"] + 0.5) * srs)
        sf.write("worst.wav", ys[a:b], srs)
        worst_audio = "worst.wav"
        worst_md = f"**{sec.capitalize()}**, {r['start']}–{r['end']}s, {r['note']} at **{r['cents']:+.0f} cents**"
    notes = rehearsal_notes_ensemble(results)
    return verdict_html(results, why), timeline(results), worst_audio, worst_md, notes, table

# ---------- tab 2: rehearsal assistant ----------
def run_assistant(audio_path, sections_text, question):
    try:
        if not audio_path:
            raise InputProblem("Upload a recording first, or click one of the demo recordings.")
        if not (question or "").strip():
            raise InputProblem("Type a question first.")
        in_piece = parse_sections(sections_text)
    except InputProblem as e:
        return f"**Can't answer yet:** {e}", "", None
    dur = round(min(librosa.get_duration(path=audio_path), 15), 1)
    plan = ask_json(
        f"You are the planning step of a rehearsal assistant. The recording is {dur} seconds long and "
        f"contains these sections: {in_piece}. The director asks: \"{question}\"\n"
        "Decide which sections to check and which time window matters. If the question refers to a note "
        "by its number (e.g. 'the third note'), use the whole clip. "
        'Reply with ONLY a JSON object like {"sections": ["cello"], "start": 0, "end": null}. '
        "Use only section names from the list; use null for start/end to mean the whole clip.")
    secs = [s for s in plan.get("sections", []) if s in in_piece] or in_piece
    start = plan.get("start") if isinstance(plan.get("start"), (int, float)) else 0
    end = plan.get("end") if isinstance(plan.get("end"), (int, float)) else dur
    start, end = max(0, min(start, dur)), max(0, min(end, dur))
    if end - start < 1:          # nonsense window: fall back to the whole clip
        start, end = 0, dur

    results_all, _, why = measure_ensemble(audio_path, in_piece)
    results, lines = {}, []
    for s in secs:
        numbered = list(enumerate(results_all[s], 1))      # note numbers count from the start of the clip
        rows = [r for k, r in numbered if r["end"] >= start and r["start"] <= end]
        results[s] = rows
        for k, r in numbered:
            if not (r["end"] >= start and r["start"] <= end): continue
            v = "in tune" if abs(r["cents"]) <= 15 else \
                f"OUT OF TUNE, {abs(r['cents']):.0f} cents {'SHARP' if r['cents'] > 0 else 'FLAT'}"
            lines.append(f"{s} held note #{k}, {r['start']}-{r['end']}s, {r['note']}: {v}")
        if not rows:
            lines.append(f"{s}: " + ("couldn't be isolated clearly" if not results_all[s]
                                     else "no held notes in this time window"))
    answer = muse(
        f"You are a rehearsal assistant for an ensemble director. They asked: \"{question}\"\n"
        f"The sections in this recording are: {in_piece}.\n"
        "Answer directly in 2-4 short sentences using ONLY these measured results. Each verdict is FINAL; "
        "do not recalculate. Summarize patterns instead of listing every note. If something is out of tune, "
        "say which section, when, sharp or flat, and one concrete fix. If the question asks about an "
        "instrument that isn't in the list, or isn't about this recording's tuning, say plainly what you "
        "can and can't tell from these results; never invent results.\n\nRESULTS:\n" + "\n".join(lines))
    if answer is None:
        answer = "*Muse Spark didn't respond, so here are the raw measurements:*\n\n" + \
                 "\n".join(f"- {l}" for l in lines)
    plan_md = (f"**Muse Spark's plan:** check {', '.join(s.capitalize() for s in secs)}, "
               f"from {start:.1f}s to {end:.1f}s  \n*How we judged: {why}*")
    empty = {s: "no held notes in this time window" if results_all[s] else
                "couldn't isolate clearly enough to judge" for s in secs}
    return plan_md, answer, timeline(results, empty)

# ---------- layout ----------
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero {text-align: center; padding: 28px 12px 8px;}
#hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;}
#hero .tag {font-size: 1.15rem; opacity: 0.8; margin-top: 6px;}
#hero .chips {margin-top: 12px;}
#hero .chip {display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 999px;
             font-size: 0.85rem; border: 1px solid rgba(127,127,127,0.35);}
.verdict {border-radius: 14px; padding: 18px 20px; border: 1px solid;}
.verdict.bad {background: rgba(229,72,77,0.08); border-color: rgba(229,72,77,0.45);}
.verdict.good {background: rgba(48,164,108,0.08); border-color: rgba(48,164,108,0.45);}
.verdict .v-head {font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.75;}
.verdict.bad .v-head {color: #e5484d;} .verdict.good .v-head {color: #30a46c;}
.verdict.warn {background: rgba(245,166,35,0.10); border-color: rgba(245,166,35,0.55);}
.verdict.warn .v-head {color: #f5a623;}
.verdict.neutral {background: rgba(139,148,158,0.10); border-color: rgba(139,148,158,0.5);}
.verdict.neutral .v-head {color: #8b949e;}
.verdict .v-line {font-size: 1.35rem; margin-top: 6px; line-height: 1.35;}
.verdict .v-note {font-size: 0.85rem; opacity: 0.7; margin-top: 10px;}
#placeholder {opacity: 0.6; padding: 18px;}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 6px;}
.step {border-radius: 14px; padding: 18px; border: 1px solid rgba(127,127,127,0.25);
       background: rgba(99,102,241,0.06);}
.step .num {width: 30px; height: 30px; border-radius: 999px; display: flex; align-items: center;
            justify-content: center; font-weight: 700; background: #4f46e5; color: white;}
.step .st {font-size: 1.15rem; font-weight: 700; margin-top: 10px;}
.step .sd {font-size: 0.95rem; opacity: 0.8; margin-top: 4px; line-height: 1.4;}
.progress-text {display: none !important;}   /* hide Gradio's elapsed/estimate timer; it overshoots */
.verdict.working .v-line::after {content: ""; display: inline-block; width: 14px; height: 14px; margin-left: 10px;
  border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; vertical-align: middle;
  animation: ens-spin 0.8s linear infinite;}
@keyframes ens-spin {to {transform: rotate(360deg);}}
.steps-hint {text-align: center; opacity: 0.7; margin-top: 14px; font-size: 0.95rem;}
@media (max-width: 700px) {.steps {grid-template-columns: 1fr;}}
"""
HERO = """
<div id="hero">
  <h1>EnsembleAI</h1>
  <div class="tag">Find out who in your ensemble is out of tune, and when.</div>
  <div class="chips">
    <span class="chip">Section isolation: Meta SAM Audio</span>
    <span class="chip">Reasoning and rehearsal notes: Muse Spark</span>
    <span class="chip">Tuning measured against the group, not a fixed A440</span>
  </div>
</div>
"""
STEPS = """
<div class="steps">
  <div class="step"><div class="num">1</div><div class="st">Isolate</div>
    <div class="sd">Meta SAM Audio pulls each section out of the mix from just its name.</div></div>
  <div class="step"><div class="num">2</div><div class="st">Measure</div>
    <div class="sd">Every held note is compared to the group's own tuning, in cents.</div></div>
  <div class="step"><div class="num">3</div><div class="st">Explain</div>
    <div class="sd">Muse Spark writes rehearsal notes and answers your questions.</div></div>
</div>
<div class="steps-hint">Pick a demo recording above or upload your own, then click <b>Analyze the ensemble</b>.
Each section takes about 20 seconds to isolate.</div>
"""
WORKING_REPORT = ("<div class='verdict neutral working'><div class='v-head'>Working</div>"
                  "<div class='v-line'>Isolating each section with Meta SAM Audio, then measuring every held note…</div>"
                  "<div class='v-note'>About 20 seconds per section the first time; reruns on the same recording are much faster.</div></div>")
WORKING_ASK = ("⏳ **Muse Spark is planning** which sections and passages to check, then EnsembleAI measures them "
               "and Muse Spark answers. *About 20 seconds per section the first time; follow-up questions are faster.*")
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"])

_blocks_kw = {} if "css" in inspect.signature(gr.Blocks.launch).parameters else {"theme": THEME, "css": CSS}
with gr.Blocks(title="EnsembleAI", **_blocks_kw) as app:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            audio_in = gr.Audio(type="filepath", label="Ensemble recording (first 15 seconds are analyzed)")
        with gr.Column(scale=2):
            sections_in = gr.Textbox(value="clarinet, violin, cello", label="Sections in the piece",
                                     info="Comma-separated. Instruments that sound different separate best.")
            demo_files = [f for f in ["cq_detuned.wav", "cq.wav"] if os.path.exists(f)]
            if demo_files:
                gr.Examples([[f] for f in demo_files], [audio_in],
                            label="Try a demo: Mozart Clarinet Quintet (cq_detuned = clarinet made 25 cents sharp)")

    with gr.Tabs():
        with gr.Tab("Ensemble report"):
            go = gr.Button("Analyze the ensemble", variant="primary", size="lg")
            verdict = gr.HTML(STEPS)
            with gr.Column(visible=False) as report_results:
              plot = gr.Plot(label="Who's out of tune, and when", show_label=False)
              with gr.Row():
                with gr.Column():
                    gr.Markdown("### Hear the problem")
                    worst_md = gr.Markdown()
                    worst_audio = gr.Audio(label="Isolated worst passage", interactive=False)
                with gr.Column():
                    gr.Markdown("### Rehearsal notes from Muse Spark")
                    notes = gr.Markdown()
              with gr.Accordion("Every held note", open=False):
                table = gr.Dataframe(headers=["Section", "Time", "Note", "Cents off", "Status"],
                                     interactive=False, wrap=True)
            go.click(lambda: WORKING_REPORT, None, verdict
                     ).then(run_report, [audio_in, sections_in], [verdict, plot, worst_audio, worst_md, notes, table]
                     ).then(lambda p: gr.update(visible=p is not None), plot, report_results)

        with gr.Tab("Ask the rehearsal assistant"):
            gr.Markdown("Ask in plain English. **Muse Spark plans** which sections and passages to check, "
                        "**runs EnsembleAI's measurement tools**, then **answers from the measured results**.")
            with gr.Row():
                question = gr.Textbox(value="Is anyone sharp?", label="Your question", scale=4)
                ask = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples([["Is anyone sharp?"], ["Just check the cello."], ["Is anyone out of tune in the last five seconds?"]],
                        [question], label="Try asking")
            plan_md = gr.Markdown()
            answer_md = gr.Markdown()
            with gr.Column(visible=False) as ask_results:
                a_plot = gr.Plot(show_label=False)
            for trigger in (ask.click, question.submit):
                trigger(lambda: (WORKING_ASK, ""), None, [plan_md, answer_md]
                        ).then(run_assistant, [audio_in, sections_in, question], [plan_md, answer_md, a_plot]
                        ).then(lambda p: gr.update(visible=p is not None), a_plot, ask_results)

_launch_kw = {"theme": THEME, "css": CSS} if "css" in inspect.signature(gr.Blocks.launch).parameters else {}
app.launch(share=True, debug=True, **_launch_kw)
