#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.1"]
# ///
"""Turn a video (URL or local file) into timestamped frames + transcript for an LLM to read.

Pipeline: yt-dlp metadata -> native subtitles (preferred) -> download <=720p ->
ffmpeg scene detection + gap filling -> near-duplicate removal -> frames (<=768px) +
3x3 contact sheets -> local mlx-whisper transcription when no subtitles -> timeline.md.

Usage:
  video_extract.py URL_OR_FILE [--probe] [--start 10:00 --end 15:00] [--max-frames 100]
                   [--no-frames] [--force-asr] [--lang zh] [--cookies-from-browser chrome]
  video_extract.py URL_OR_FILE --at 03:12,05:40      # grab full-res frames at given times
  video_extract.py URL_OR_FILE --audio-only [--audio-format wav --sample-rate 16000 --mono]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "clipsight"
HERE = Path(__file__).resolve().parent
PREFERRED_SUB_LANGS = ["zh-TW", "zh-Hant", "zh-HK", "zh", "zh-Hans", "zh-CN", "en", "en-US", "en-GB", "ja", "ko"]
SCENE_THRESHOLD = 0.3
FRAME_MAX_EDGE = 768
AT_FRAME_MAX_EDGE = 1280
SHEET_GRID = 3
SHEET_CELL_W = 512  # max cell edge
SHEET_GAP = 6
DEDUP_THRESHOLD = 3.0  # mean abs difference (0-255) of 16x16 grayscale thumbnails
DEDUP_MAX_GAP = 60.0  # keep at least one frame per minute even on a static shot

# Run yt-dlp on uv's own Python instead of the Homebrew one: Homebrew Python upgrades
# can break yt-dlp (e.g. python@3.15 pyexpat vs system libexpat). Override with YTDLP="yt-dlp".
YTDLP = os.environ.get("YTDLP", "uvx --python 3.13 --from yt-dlp[default]@latest yt-dlp").split()


def log(msg: str) -> None:
    print(f"[clipsight] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_time(s: str) -> float:
    """'90', '1:30', '01:01:30', '1:30.5' -> seconds."""
    try:
        sec = 0.0
        for part in s.strip().split(":"):
            sec = sec * 60 + float(part)
    except ValueError:
        sec = -1.0
    if not (0 <= sec < float("inf")):  # also rejects nan
        sys.exit(f"bad time: {s!r} (use SS, MM:SS or HH:MM:SS)")
    return sec


def range_tag(start: float | None, end: float | None) -> str:
    """Cache/file suffix for a time range, e.g. '_r10.5-20' ('' when no range)."""
    if start is None and end is None:
        return ""
    return f"_r{start or 0:g}-{'end' if end is None else f'{end:g}'}"


# ---------------------------------------------------------------- metadata

def ytdlp_base(args) -> list[str]:
    cmd = YTDLP + ["--no-playlist", "--no-warnings"]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    return cmd


def probe_remote(url: str, args) -> dict:
    p = run(ytdlp_base(args) + ["-J", "--", url])
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "unknown error"
        hint = ""
        if re.search(r"sign in|login|private|members|age", err, re.I):
            hint = " (needs login: retry with --cookies-from-browser chrome)"
        sys.exit(f"yt-dlp failed: {err}{hint}")
    info = json.loads(p.stdout)
    if info.get("_type") == "playlist":
        sys.exit("URL is a playlist; pass a single video URL.")
    return info


def pick_subtitle(info: dict, want_lang: str | None) -> tuple[str, bool] | None:
    """Return (lang_key, is_auto). Manual subs first; auto captions only in the original language."""
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    orig = (info.get("language") or "").split("-")[0] or None
    order = ([want_lang] if want_lang else []) + ([orig] if orig else []) + PREFERRED_SUB_LANGS
    for lang in order:
        for k in manual:
            if k == lang or k.split("-")[0] == lang:
                return k, False
    # Auto captions: YouTube exposes machine translations into every language; only the
    # "-orig" track (or the original language) is real ASR of the audio.
    for k in auto:
        if k.endswith("-orig"):
            return k, True
    for lang in ([orig] if orig else []):
        if lang in auto:
            return lang, True
    return None


def summarize_probe(info: dict) -> dict:
    size = None
    fmts = [f for f in info.get("formats") or [] if (f.get("height") or 0) <= 720]
    sizes = [f.get("filesize") or f.get("filesize_approx") for f in fmts]
    sizes = [s for s in sizes if s]
    if sizes:
        size = round(max(sizes) / 1e6, 1)
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration_sec": info.get("duration"),
        "duration": ts(info.get("duration") or 0),
        "approx_mb_720p": size,
        "manual_subs": sorted((info.get("subtitles") or {}).keys()),
        "has_auto_captions": bool(info.get("automatic_captions")),
        "chosen_subtitle": pick_subtitle(info, None),
        "chapters": [f"{ts(c['start_time'])} {c['title']}" for c in info.get("chapters") or []],
    }


# ---------------------------------------------------------------- subtitles

def parse_json3(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = []
    for ev in data.get("events", []):
        if "segs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if text:
            segs.append({"start": ev.get("tStartMs", 0) / 1000, "text": text})
    return segs


def parse_vtt(path: Path) -> list[dict]:
    segs, prev_lines = [], set()
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace"))
    for b in blocks:
        m = re.search(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s+-->", b) or re.search(r"(\d{2}):(\d{2})[.,](\d{3})\s+-->", b)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000 if len(g) == 4 else g[0] * 60 + g[1] + g[2] / 1000
        lines = [re.sub(r"<[^>]+>", "", l).strip() for l in b.split("\n") if "-->" not in l]
        lines = [l for l in lines if l and not l.isdigit()]
        # Rolling auto-captions repeat the previous cue's lines; keep only new ones.
        new = [l for l in lines if l not in prev_lines]
        prev_lines = set(lines)
        if new:
            segs.append({"start": start, "text": " ".join(new)})
    return segs


def fetch_subtitles(url: str, lang: str, is_auto: bool, out: Path, args) -> list[dict]:
    flag = "--write-auto-subs" if is_auto else "--write-subs"
    cmd = ytdlp_base(args) + ["--skip-download", flag, "--sub-langs", lang,
                              "--sub-format", "json3/vtt/best", "-o", str(out / "subs.%(ext)s"), "--", url]
    p = run(cmd)
    files = sorted(out.glob("subs.*"))
    if p.returncode != 0 or not files:
        log(f"subtitle download failed: {p.stderr.strip()[-300:]}")
        return []
    f = files[0]
    segs = parse_json3(f) if f.suffix == ".json3" else parse_vtt(f)
    f.unlink()
    return segs


# ---------------------------------------------------------------- media

def get_video(source: str, is_local: bool, vid: str, args) -> tuple[Path, bool]:
    """Return (video path, downloaded by this call). Local files are used in place; remote videos
    are downloaded once to the per-video cache dir. Only a finished download is named video.*,
    so fragments of an interrupted run are never mistaken for the video."""
    if is_local:
        return Path(source).expanduser().resolve(), False
    vdir = CACHE_ROOT / vid
    vdir.mkdir(parents=True, exist_ok=True)
    have = sorted(vdir.glob("video.*"))
    if have and not args.refresh:
        return have[0], False
    for f in have + sorted(vdir.glob("dl.*")):
        f.unlink()
    cmd = ytdlp_base(args) + [
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4", "-o", str(vdir / "dl.%(ext)s"),
        "--no-simulate", "--print", "after_move:filepath", "--", source]
    log("downloading video (<=720p)...")
    p = run(cmd)
    done = Path(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else None
    if done is None or not done.exists():
        sys.exit(f"download failed: {p.stderr.strip()[-500:]}\n"
                 "Possibly deleted/private/region-blocked. Try --cookies-from-browser chrome.")
    video = done.rename(vdir / f"video{done.suffix}")
    return video, True


def media_info(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)])
    d = json.loads(p.stdout or "{}")
    types = {s.get("codec_type") for s in d.get("streams", [])}
    return {"duration": float(d.get("format", {}).get("duration") or 0),
            "has_video": "video" in types, "has_audio": "audio" in types}


def window_args(start: float, end: float) -> list[str]:
    """ffmpeg input options limiting decoding to [start, end); output timestamps restart at 0."""
    return ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]


def detect_scenes(video: Path, start: float, end: float) -> list[tuple[float, float]]:
    """Return [(absolute time, scene_score)] for scene cuts, decoding a downscaled stream."""
    log("detecting scene changes...")
    p = run(["ffmpeg", "-hide_banner", "-nostats", *window_args(start, end), "-i", str(video), "-an",
             "-vf", f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',metadata=print",
             "-f", "null", "-"])
    scenes, t = [], None
    for line in p.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"lavfi\.scene_score=([\d.]+)", line)
        if m and t is not None:
            scenes.append((start + t, float(m.group(1))))
            t = None
    return scenes


def thin(frames: list[dict], start: float, end: float, n: int) -> list[dict]:
    """Keep coverage: one frame per equal time bucket, preferring the strongest scene cut."""
    if len(frames) <= n:
        return frames
    span = max(end - start, 1e-6)
    buckets: dict[int, dict] = {}
    for c in frames:
        k = min(max(int((c["t"] - start) / span * n), 0), n - 1)
        if k not in buckets or c["score"] > buckets[k]["score"]:
            buckets[k] = c
    return [buckets[k] for k in sorted(buckets)]


def plan_frames(start: float, end: float, scenes: list[tuple[float, float]], interval: float, max_frames: int) -> list[dict]:
    first = start + 0.5 if end - start > 1 else start
    last = max(end - 0.5, first)
    cands = [{"t": first, "kind": "start", "score": 2.0}]
    cands += [{"t": min(t + 0.2, last), "kind": "scene", "score": s} for t, s in scenes]
    cands.sort(key=lambda c: c["t"])
    # Fill long stretches without a cut so no gap exceeds `interval`.
    filled, edges = [], cands + [{"t": last, "kind": "end", "score": 0.0}]
    for a, b in zip(edges, edges[1:]):
        filled.append(a)
        gap = b["t"] - a["t"]
        n = int(gap // interval)
        for i in range(1, n + 1):
            filled.append({"t": a["t"] + gap * i / (n + 1), "kind": "fill", "score": 0.0})
    # Dedupe frames closer than 1s.
    frames = []
    for c in filled:
        if not frames or c["t"] - frames[-1]["t"] >= 1.0:
            frames.append(c)
        elif c["score"] > frames[-1]["score"]:
            frames[-1] = c
    return thin(frames, start, end, max_frames)


def grab_frame(video: Path, t: float, dest: Path, max_edge: int) -> bool:
    scale = f"scale='min({max_edge},iw)':'min({max_edge},ih)':force_original_aspect_ratio=decrease"
    p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}",
             "-i", str(video), "-frames:v", "1", "-vf", scale, "-q:v", "3", str(dest)])
    return p.returncode == 0 and dest.exists()


def extract_frames(video: Path, frames: list[dict], out: Path) -> None:
    fdir = out / "frames"
    fdir.mkdir(exist_ok=True)

    def grab(f):
        name = f"t{f['t']:09.2f}.jpg"
        f["file"] = f"frames/{name}" if grab_frame(video, f["t"], fdir / name, FRAME_MAX_EDGE) else None

    log(f"extracting {len(frames)} frames...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(grab, frames))
    frames[:] = [f for f in frames if f.get("file")]


def drop_near_duplicates(frames: list[dict], out: Path) -> int:
    """Remove frames that look almost identical to the previous kept one (static slides, talking heads)."""
    from PIL import Image

    def sig(f):
        return Image.open(out / f["file"]).convert("L").resize((16, 16)).tobytes()

    kept, prev, dropped = [], None, 0
    for f in frames:
        s = sig(f)
        if prev is not None and f["kind"] != "start" and f["t"] - kept[-1]["t"] < DEDUP_MAX_GAP:
            diff = sum(abs(a - b) for a, b in zip(s, prev)) / len(s)
            if diff < DEDUP_THRESHOLD:
                (out / f["file"]).unlink(missing_ok=True)
                dropped += 1
                continue
        kept.append(f)
        prev = s
    frames[:] = kept
    return dropped


def finalize_frame_names(frames: list[dict], out: Path) -> None:
    """Renumber kept frames as fNNN_HH-MM-SS.jpg in time order."""
    for i, f in enumerate(frames, 1):
        name = f"f{i:03d}_{ts(f['t']).replace(':', '-')}.jpg"
        (out / f["file"]).rename(out / "frames" / name)
        f["file"] = f"frames/{name}"


def make_sheets(frames: list[dict], out: Path) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont

    sdir = out / "sheets"
    sdir.mkdir(exist_ok=True)
    per = SHEET_GRID * SHEET_GRID
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    sheets = []
    for si in range(0, len(frames), per):
        group = frames[si:si + per]
        thumbs = []
        for f in group:
            im = Image.open(out / f["file"]).convert("RGB")
            im.thumbnail((SHEET_CELL_W, SHEET_CELL_W))
            thumbs.append((f, im))
        # Every cell has the same size; landscape frames make short cells, portrait ones narrow cells.
        cell_w = max(im.width for _, im in thumbs)
        cell_h = max(im.height for _, im in thumbs)
        rows = (len(thumbs) + SHEET_GRID - 1) // SHEET_GRID
        pitch_x, pitch_y = cell_w + SHEET_GAP, cell_h + SHEET_GAP
        sheet = Image.new("RGB", (SHEET_GRID * pitch_x - SHEET_GAP, rows * pitch_y - SHEET_GAP), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (f, im) in enumerate(thumbs):
            x, y = (j % SHEET_GRID) * pitch_x, (j // SHEET_GRID) * pitch_y
            sheet.paste(im, (x + (cell_w - im.width) // 2, y + (cell_h - im.height) // 2))
            label = ts(f["t"])
            box = draw.textbbox((x + 6, y + 6), label, font=font)
            draw.rectangle((box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4), fill="black")
            draw.text((x + 6, y + 6), label, fill="yellow", font=font)
        name = f"sheet{si // per + 1:02d}_{ts(group[0]['t']).replace(':', '-')}_to_{ts(group[-1]['t']).replace(':', '-')}.jpg"
        sheet.save(sdir / name, quality=85)
        sheets.append({"file": f"sheets/{name}", "from": ts(group[0]["t"]), "to": ts(group[-1]["t"])})
    return sheets


def transcribe_local(video: Path, out: Path, lang: str | None, start: float, end: float) -> list[dict]:
    audio = out / "audio.mp3"
    p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *window_args(start, end), "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(audio)])
    if p.returncode != 0:
        log(f"audio extraction failed: {p.stderr.strip()[-300:]}")
        return []
    log("transcribing locally with mlx-whisper (first run downloads the model)...")
    seg_json = out / "asr.json"
    cmd = ["uv", "run", "--script", str(HERE / "transcribe_mlx.py"), str(audio), str(seg_json)]
    if lang:
        cmd += ["--lang", lang]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=sys.stderr)
    audio.unlink(missing_ok=True)
    if p.returncode != 0 or not seg_json.exists():
        log("transcription failed")
        return []
    segs = json.loads(seg_json.read_text(encoding="utf-8"))
    seg_json.unlink()
    for s in segs:
        s["start"] += start
        s["end"] += start
    return segs


# ---------------------------------------------------------------- output

def write_outputs(out: Path, meta: dict, frames: list[dict], sheets: list[dict], segs: list[dict]) -> None:
    (out / "transcript.md").write_text(
        "\n".join(f"[{ts(s['start'])}] {s['text']}" for s in segs) + "\n", encoding="utf-8")

    L = [f"# {meta.get('title') or 'Untitled'}", ""]
    for k in ("source", "uploader", "upload_date", "duration", "range", "transcript_source"):
        if meta.get(k):
            L.append(f"- **{k}**: {meta[k]}")
    L.append(f"- **frames**: {len(frames)} (scene cuts + gap fill, near-duplicates removed, max edge {FRAME_MAX_EDGE}px)")
    if meta.get("limitations"):
        L.append(f"- **limitations**: {'; '.join(meta['limitations'])}")
    if meta.get("chapters"):
        L += ["", "## Chapters"] + [f"- [{ts(c['start_time'])}] {c['title']}" for c in meta["chapters"]]
    if meta.get("description"):
        L += ["", "## Description", meta["description"][:1500]]
    if sheets:
        L += ["", "## Contact sheets (3x3, timestamp in yellow; read these first)"]
        L += [f"- {s['file']}  ({s['from']} - {s['to']})" for s in sheets]

    L += ["", "## Timeline"]
    if frames:
        # Each frame gets the speech until the next frame; speech before the first frame goes to it too.
        bounds = [f["t"] for f in frames] + [float("inf")]
        si = 0
        for i, f in enumerate(frames):
            chunk = []
            while si < len(segs) and segs[si]["start"] < bounds[i + 1]:
                chunk.append(segs[si]["text"])
                si += 1
            L += ["", f"### [{ts(f['t'])}] {f['file']} ({f['kind']})"]
            if chunk:
                L.append(" ".join(chunk))
    else:
        L += [f"[{ts(s['start'])}] {s['text']}" for s in segs]
    (out / "timeline.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    meta = dict(meta, frames=frames, sheets=sheets, segments=len(segs))
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- modes

LOCAL_COPY_EXT = {"aac": "m4a", "alac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg", "flac": "flac"}


def extract_audio(source: str, is_local: bool, info: dict, vid: str, args) -> None:
    """Save the audio track (no transcription) for other tools; print its path on stdout."""
    out = Path(args.audio_dir).expanduser() if args.audio_dir else CACHE_ROOT / vid / "audio"
    out.mkdir(parents=True, exist_ok=True)
    fmt = args.audio_format
    rng = args.start is not None or args.end is not None
    stem = vid + range_tag(args.start, args.end)
    src = Path(source).expanduser() if is_local else None
    if not is_local:
        cmd = ytdlp_base(args) + ["-f", "ba/b", "-x", "--audio-format", "best" if fmt == "original" else fmt,
                                  "-o", str(out / f"{vid}.full.%(ext)s")]
        if args.ffmpeg_audio_opts:
            cmd += ["--postprocessor-args", "ExtractAudio:" + " ".join(args.ffmpeg_audio_opts)]
        log("downloading audio...")
        p = run(cmd + ["--", source])
        files = [f for f in out.glob(f"{vid}.full.*") if f.suffix not in (".part", ".ytdl", ".json")]
        if p.returncode != 0 or not files:
            sys.exit(f"audio download failed: {p.stderr.strip()[-500:]}")
        downloaded = max(files, key=lambda f: f.stat().st_mtime)
        if rng:
            src = downloaded
        else:
            dest = out / f"{stem}{downloaded.suffix}"
            downloaded.replace(dest)
    if src is not None:
        p = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                 "-of", "csv=p=0", str(src)])
        codec = p.stdout.strip()
        if not codec:
            sys.exit("no audio track in this file")
        # Remote audio was already converted by yt-dlp, so only cut it; local audio still needs encoding.
        copy = fmt == "original" or not is_local
        ext = LOCAL_COPY_EXT.get(codec, "mka") if fmt == "original" else fmt
        dest = out / f"{stem}.{ext}"
        start = args.start or 0.0
        win = window_args(start, args.end) if args.end is not None else ["-ss", f"{start:.3f}"]
        p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *win, "-i", str(src), "-vn", "-map", "a:0",
                 *(["-c:a", "copy"] if copy else args.ffmpeg_audio_opts), str(dest)])
        if not is_local:
            src.unlink(missing_ok=True)
        if p.returncode != 0:
            sys.exit(f"audio extraction failed: {p.stderr.strip()[-300:]}")
    sidecar = {"source": source, "audio": str(dest), **media_info(dest)}
    if rng:
        sidecar["range"] = [args.start or 0, args.end]
    if info:
        sidecar.update({k: info.get(k) for k in ("id", "title", "uploader", "upload_date", "description", "webpage_url")})
    dest.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"audio saved ({sidecar['duration']:.1f}s)")
    print(dest)


def grab_at(source: str, is_local: bool, vid: str, times: list[float], args) -> None:
    """Full-resolution frames at specific times; the downloaded video stays cached for later calls."""
    video, _ = get_video(source, is_local, vid, args)
    out = CACHE_ROOT / vid / "at"
    out.mkdir(parents=True, exist_ok=True)
    duration = media_info(video)["duration"]
    for t in times:
        if duration and t >= duration:
            log(f"skip {ts(t)}: past the end ({ts(duration)})")
            continue
        dest = out / f"at_{ts(t).replace(':', '-')}_{t:.1f}.jpg"
        if grab_frame(video, t, dest, AT_FRAME_MAX_EDGE):
            print(dest)
        else:
            log(f"could not grab frame at {ts(t)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video URL or local file path")
    ap.add_argument("--probe", action="store_true", help="print metadata only, no download")
    ap.add_argument("--start", help="only analyze from this time (SS, MM:SS, HH:MM:SS)")
    ap.add_argument("--end", help="only analyze up to this time")
    ap.add_argument("--at", help="comma-separated times: save full-res frames there and exit")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--interval", type=float, default=5.0, help="max seconds between frames before capping")
    ap.add_argument("--no-dedup", action="store_true", help="keep near-identical frames")
    ap.add_argument("--no-frames", action="store_true", help="transcript only")
    ap.add_argument("--force-asr", action="store_true", help="ignore subtitles, transcribe audio locally")
    ap.add_argument("--lang", help="subtitle / ASR language hint, e.g. zh, en, ja")
    ap.add_argument("--cookies-from-browser", help="e.g. chrome, safari (for login-only videos)")
    ap.add_argument("--keep-video", action="store_true", help="keep the downloaded video in the cache")
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    ap.add_argument("--audio-only", action="store_true", help="just save the audio file (prints its path)")
    ap.add_argument("--audio-format", default="original", choices=["original", "wav", "flac", "mp3", "m4a"],
                    help="original = no re-encode (best quality)")
    ap.add_argument("--sample-rate", type=int, help="resample (re-encoded formats only), e.g. 16000")
    ap.add_argument("--mono", action="store_true", help="downmix to mono (re-encoded formats only)")
    ap.add_argument("--audio-dir", help="where to save audio (default: cache dir)")
    args = ap.parse_args()
    args.ffmpeg_audio_opts = []
    if args.sample_rate or args.mono:
        if args.audio_format == "original":
            sys.exit("--sample-rate/--mono need a re-encoded --audio-format (wav/flac/mp3/m4a)")
        args.ffmpeg_audio_opts = (["-ar", str(args.sample_rate)] if args.sample_rate else []) + (["-ac", "1"] if args.mono else [])
    args.start = parse_time(args.start) if args.start else None
    args.end = parse_time(args.end) if args.end else None
    if args.start is not None and args.end is not None and args.end <= args.start:
        sys.exit("--end must be after --start")

    for tool in ("ffmpeg", "ffprobe", "uv", YTDLP[0]):
        if not shutil.which(tool):
            sys.exit(f"missing dependency: {tool} (brew install {tool})")

    local = Path(args.source).expanduser()
    is_local = local.is_file()
    info: dict = {}
    if is_local:
        st = local.resolve().stat()
        vid = "local-" + hashlib.sha1(f"{local.resolve()}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:12]
        if args.probe:
            print(json.dumps({"id": vid, "title": local.name, **media_info(local),
                              "size_mb": round(st.st_size / 1e6, 1)}, ensure_ascii=False, indent=2))
            return
    else:
        info = probe_remote(args.source, args)
        if args.probe:
            print(json.dumps(summarize_probe(info), ensure_ascii=False, indent=2))
            return
        vid = re.sub(r"[^\w-]", "_", f"{info.get('extractor_key', 'web')}-{info.get('id')}")

    if args.audio_only:
        extract_audio(args.source, is_local, info, vid, args)
        return
    if args.at:
        grab_at(args.source, is_local, vid, [parse_time(t) for t in args.at.split(",") if t.strip()], args)
        return

    rng = range_tag(args.start, args.end)
    variant = (("noframes" if args.no_frames else f"f{args.max_frames}i{args.interval:g}")
               + ("-nodedup" if args.no_dedup else "") + ("-asr" if args.force_asr else "")
               + (f"-{args.lang}" if args.lang else "") + rng)
    out = CACHE_ROOT / vid / variant
    if (out / "timeline.md").exists() and not args.refresh:
        log("cache hit")
        if args.keep_video and not is_local:
            get_video(args.source, is_local, vid, args)  # make sure the video is there to keep
        print(out / "timeline.md")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    meta = {"source": args.source, "limitations": []}
    segs: list[dict] = []
    if is_local:
        meta["title"] = local.name
    else:
        meta.update(title=info.get("title"), uploader=info.get("uploader"),
                    upload_date=info.get("upload_date"), description=info.get("description"),
                    chapters=info.get("chapters"))
        sub = None if args.force_asr else pick_subtitle(info, args.lang)
        if sub:
            log(f"fetching {'auto' if sub[1] else 'manual'} subtitles: {sub[0]}")
            segs = fetch_subtitles(args.source, sub[0], sub[1], out, args)
            if segs:
                meta["transcript_source"] = f"{'YouTube auto-captions' if sub[1] else 'uploaded subtitles'} ({sub[0]})"

    start = args.start or 0.0
    end = args.end if args.end is not None else (float(info.get("duration") or 0) or float("inf"))
    need_media = not args.no_frames or not segs
    video, downloaded, mi = None, False, {}
    if need_media:
        video, downloaded = get_video(args.source, is_local, vid, args)
        mi = media_info(video)
        if mi["duration"]:
            end = min(end, mi["duration"])
        if end == float("inf"):
            sys.exit("cannot tell the video's duration; pass --end")
        meta["duration"] = ts(mi["duration"])
        if not segs:
            if mi["has_audio"]:
                segs = transcribe_local(video, out, args.lang, start, end)
                if segs:
                    meta["transcript_source"] = "local Whisper ASR (mlx-whisper large-v3-turbo; may contain recognition errors)"
                else:
                    meta["limitations"].append("speech recognition failed; answer from frames only")
            else:
                meta["limitations"].append("video has no audio track")
    else:
        meta["duration"] = ts(info.get("duration") or 0)
    if start >= end:
        sys.exit(f"--start is past the end of the video ({meta['duration']})")
    if rng:
        meta["range"] = f"{ts(start)} - {ts(end) if end != float('inf') else 'end'} (only this part was analyzed)"
        # Keep cues that overlap the range; a cue lasts until the next one starts when it has no end.
        nxt = [s["start"] for s in segs[1:]] + [float("inf")]
        segs = [s for s, n in zip(segs, nxt) if s["start"] < end and s.get("end", n) > start]

    frames, sheets = [], []
    if not args.no_frames and mi.get("has_video"):
        frames = plan_frames(start, end, detect_scenes(video, start, end), args.interval,
                             args.max_frames if args.no_dedup else args.max_frames * 2)
        extract_frames(video, frames, out)
        if not args.no_dedup:
            log(f"removed {drop_near_duplicates(frames, out)} near-duplicate frames")
            keep = thin(frames, start, end, args.max_frames)
            keep_ids = {id(f) for f in keep}
            for f in frames:
                if id(f) not in keep_ids:
                    (out / f["file"]).unlink(missing_ok=True)
            frames = keep
        finalize_frame_names(frames, out)
        sheets = make_sheets(frames, out)
        gap = (end - start) / max(len(frames), 1)
        if gap > args.interval * 1.5:
            meta["limitations"].append(f"~1 frame per {gap:.0f}s on average (sparse: long span or static shots)")

    # Only remove a video this run downloaded; one cached by --at / --keep-video stays.
    if downloaded and not args.keep_video:
        video.unlink(missing_ok=True)

    write_outputs(out, meta, frames, sheets, segs)
    log(f"done: {len(frames)} frames, {len(sheets)} sheets, {len(segs)} transcript segments")
    print(out / "timeline.md")


if __name__ == "__main__":
    main()
