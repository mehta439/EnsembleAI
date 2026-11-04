# EnsembleAI Test Log

Every test we ran before submission (Nov 2026), what happened, and how each problem was resolved.

## Accuracy tests

We injected known tuning errors into real recordings and checked whether EnsembleAI found them. Sections: clarinet, violin, cello unless noted.

| Test | Injected | Final result | Status |
|---|---|---|---|
| Mozart String Quartet, cello only | Cello +25 cents | Cello read +18 to +25 and was flagged | Pass |
| Clarinet Quintet, original (`cq.wav`) | None | All three sections in tune | Pass |
| Clarinet Quintet, detuned (`cq_detuned.wav`) | Clarinet +25 cents | Clarinet +22 on 9 of 12 notes; cello in tune; violin "couldn't isolate" | Pass |
| Real duet (euphonium + bassoon), solo mode | Third note known sharp | Third note (C3) ranked worst at +33 cents; Muse Spark listed it first | Pass |
| Whole quintet shifted together | Everyone +30 cents | In tune (the group is together) | Pass |
| Two sections off in opposite directions | Clarinet +25, cello −25 | Clarinet +23 flagged; cello couldn't be isolated | Partial |
| Very large detune | Clarinet +60 cents | Clarinet flagged as about −39; violin still shows a false alarm | Partial |

## Problems found and how they were resolved

| Test | What went wrong | Fix | Status |
|---|---|---|---|
| String quartet, all four sections | Viola and violin separated into mush | Switched the demo to the Clarinet Quintet (contrasting instruments) | Resolved |
| Clarinet quintet, first ensemble run | Loud clarinet leaked into every track, so the reference moved with it; in-tune cello flagged −26 | "Odd one out" reference: the section that disagrees most is set aside | Resolved |
| Reasoning line on `cq_detuned` | Said "violin disagrees most" because the muddy violin track picked the reference | Unclear sections excluded from choosing the reference | Resolved |
| Rehearsal notes (Muse Spark) | Muse Spark reported a +22 note as in tune by re-doing the math | It now only receives final verdicts, never raw numbers | Resolved |
| Duet, "bassoon, tuba" | SAM Audio didn't recognize the euphonium; opening solo silent in both tracks | Solo mode: list one section and it measures the recording directly | Resolved |
| Assistant: "the third note" | The model couldn't tell which note was third | Notes are numbered for Muse Spark | Resolved |
| Instrument not in the recording ("trumpet") | None: showed "couldn't isolate clearly" | — | Pass |
| Not music / no clear instrument | Said "couldn't isolate" but still showed "In tune" | Gray "No verdict" card when nothing can be judged | Resolved |
| Typo instrument ("trt") | Treated as a real section and said "In tune" | Checked against known instruments; friendly "isn't an instrument EnsembleAI knows" card | Resolved |
| Empty sections box / no audio | Red "Error" badges on every output | One amber "Can't analyze yet" card with the reason, in both tabs | Resolved |
| Assistant: "What's the weather?" | None: answered "I can't tell the weather from these results" | — | Pass |
| Chart with an empty lane | Labels overlapped the axis numbers | Padding added; cents labels only on wide-enough bars | Resolved |
| Chart text | Gray text unreadable on white | Black, bold section names | Resolved |
| Leakage in the two-sections and large-detune tests | Clarinet's notes copied into violin/cello tracks and blamed on them | Identical out-of-tune notes credited only to the loudest track; one stray note no longer counts as a pattern | Resolved (two sections), improved (large detune) |
| First leakage fix on `cq.wav` | Wiped the clarinet lane on the in-tune recording | Leakage check limited to out-of-tune notes, so in-tune recordings are untouched | Resolved |
| Very short clips, fast passages | None: "can't judge" | — | Pass |
| File types (mp3, m4a, wav, long files) | None: first 15 seconds analyzed | — | Pass |
| Same file run twice | None: second run fast (separations reused) | — | Pass |
| Muse Spark API down (simulated) | Would have crashed | Measurements still shown, with a note that the rehearsal notes are unavailable | Resolved |
| Free Colab RAM | Crashed loading SAM Audio | Skipped unused scoring models; weights loaded straight to the GPU | Resolved |

## Known limitations

- **More than half a step off:** a +60-cent note reads as about −40 from the next note up. Every tuner has this limit.
- **Several sections off at once:** the loudest culprit is caught reliably; a second one can be missed if it doesn't separate cleanly.
- **Similar instruments and rare names:** SAM Audio blurs viola and violin and didn't recognize "euphonium"; use contrasting instruments or solo mode.
