# Run in Colab AFTER ensembleai_core.py and ensembleai_app.py (stop the running app cell first).
# Computes real results for each demo clip with the full pipeline and zips them for the hosted demo.
import json, shutil, os, subprocess

DEMOS = [
    {"id": "clarinet_detuned",
     "title": "Mozart Clarinet Quintet: clarinet made 25 cents sharp",
     "file": "cq_detuned.wav", "sections": "clarinet, violin, cello",
     "questions": ["Is anyone sharp?", "Just check the cello.", "Is anyone out of tune in the last five seconds?"]},
    {"id": "clarinet_original",
     "title": "Mozart Clarinet Quintet: original recording (in tune)",
     "file": "cq.wav", "sections": "clarinet, violin, cello",
     "questions": ["Is anyone sharp?", "Is the clarinet in tune with the strings?"]},
    {"id": "duet",
     "title": "Real recording: euphonium solo opening (third note known to be sharp)",
     "file": "duet.m4a",          # <-- change to your duet file's name in the Files panel
     "sections": "euphonium",
     "questions": ["Is the euphonium sharp on the third note?", "Which note should the player fix first?"]},
]

shutil.rmtree("demo_results", ignore_errors=True)
os.makedirs("demo_results")
manifest = []
for d in DEMOS:
    if not os.path.exists(d["file"]):
        print("SKIPPING, file not found:", d["file"]); continue
    out = f"demo_results/{d['id']}"; os.makedirs(out)
    if not d["file"].lower().endswith(".wav"):          # m4a/mp3 -> wav so every library can read it
        wav = os.path.splitext(d["file"])[0] + "_conv.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", d["file"], wav], check=True)
        d["file"] = wav
    y, sr = librosa.load(d["file"], sr=None, duration=15)
    sf.write(f"{out}/input.wav", y, sr)
    verdict, fig, worst_audio, worst_md, notes, table = run_report(d["file"], d["sections"])
    fig.savefig(f"{out}/timeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    if worst_audio:
        shutil.copy(worst_audio, f"{out}/worst.wav")
    qa = []
    for i, q in enumerate(d["questions"]):
        plan, ans, f2 = run_assistant(d["file"], d["sections"], q)
        f2.savefig(f"{out}/qa{i}.png", dpi=150, bbox_inches="tight"); plt.close(f2)
        qa.append({"question": q, "plan": plan, "answer": ans, "plot": f"qa{i}.png"})
        print("  asked:", q)
    json.dump({"id": d["id"], "title": d["title"], "sections": d["sections"], "verdict": verdict,
               "worst_md": worst_md, "has_worst": bool(worst_audio), "notes": notes,
               "table": [[c if isinstance(c, str) else float(c) for c in row] for row in (table or [])],
               "qa": qa}, open(f"{out}/result.json", "w"), indent=1)
    manifest.append({"id": d["id"], "title": d["title"]})
    print("done:", d["title"])
json.dump(manifest, open("demo_results/manifest.json", "w"), indent=1)
shutil.make_archive("demo_results", "zip", "demo_results")
print("\nDownload demo_results.zip from the Files panel.")
