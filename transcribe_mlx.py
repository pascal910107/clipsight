#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mlx-whisper"]
# ///
"""Local Whisper transcription on Apple Silicon. Writes [{start, end, text}] JSON."""
import argparse
import json

import mlx_whisper

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--lang")
ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
a = ap.parse_args()

kw = {"path_or_hf_repo": a.model, "condition_on_previous_text": False}
if a.lang:
    kw["language"] = a.lang
if a.lang == "zh":
    # Nudges Chinese output toward Traditional characters with punctuation.
    kw["initial_prompt"] = "以下是繁體中文的逐字稿，包含標點符號。"
r = mlx_whisper.transcribe(a.audio, **kw)
segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in r["segments"] if s["text"].strip()]
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(segs, f, ensure_ascii=False)
