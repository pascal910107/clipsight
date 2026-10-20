#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.1"]
# ///
"""Turn a video (URL or local file) into timestamped frames + transcript for an LLM to read.

Pipeline: yt-dlp metadata -> native subtitles (preferred) -> download <=720p ->
ffmpeg scene detection + gap filling -> frames (<=768px) + 3x3 contact sheets ->
local mlx-whisper transcription when no subtitles -> timeline.md.

Usage:
  video_extract.py URL_OR_FILE [--probe] [--max-frames 100] [--interval 5]
                   [--no-frames] [--force-asr] [--lang zh] [--cookies-from-browser chrome]
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
SHEET_GRID = 3
SHEET_CELL_W = 512  # max cell edge
SHEET_GAP = 6


def log(msg: str) -> None:
    print(f"[clipsight] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- metadata

# Run yt-dlp on uv's own Python instead of the Homebrew one: Homebrew Python upgrades
# can break yt-dlp (e.g. python@3.15 pyexpat vs system libexpat). Override with YTDLP="yt-dlp".
YTDLP = os.environ.get("YTDLP", "uvx --python 3.13 --from yt-dlp[default]@latest yt-dlp").split()


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

def download_video(url: str, out: Path, args) -> Path:
    cmd = ytdlp_base(args) + [
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4", "-o", str(out / "video.%(ext)s"), "--", url]
    log("downloading video (<=720p)...")
    p = run(cmd)
    vids = [f for f in out.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if p.returncode != 0 or not vids:
        sys.exit(f"download failed: {p.stderr.strip()[-500:]}\n"
                 "Possibly deleted/private/region-blocked. Try --cookies-from-browser chrome.")
    return vids[0]


def media_info(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)])
    d = json.loads(p.stdout or "{}")
    types = {s.get("codec_type") for s in d.get("streams", [])}
    return {"duration": float(d.get("format", {}).get("duration") or 0),
            "has_video": "video" in types, "has_audio": "audio" in types}


def detect_scenes(video: Path) -> list[tuple[float, float]]:
    """Return [(time, scene_score)] for scene cuts, decoding a downscaled stream."""
    log("detecting scene changes...")
    p = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-an",
             "-vf", f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',metadata=print",
             "-f", "null", "-"])
    out, scenes, t = p.stderr, [], None
    for line in out.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"lavfi\.scene_score=([\d.]+)", line)
        if m and t is not None:
            scenes.append((t, float(m.group(1))))
            t = None
    return scenes


def plan_frames(duration: float, scenes: list[tuple[float, float]], interval: float, max_frames: int) -> list[dict]:
    end = max(duration - 0.5, 0)
    cands = [{"t": 0.5 if duration > 1 else 0, "kind": "start", "score": 2.0}]
    cands += [{"t": min(t + 0.2, end), "kind": "scene", "score": s} for t, s in scenes]
    cands.sort(key=lambda c: c["t"])
    # Fill long stretches without a cut so no gap exceeds `interval`.
    filled, edges = [], cands + [{"t": end, "kind": "end", "score": 0.0}]
    for a, b in zip(edges, edges[1:]):
        filled.append(a)
        gap = b["t"] - a["t"]
        n = int(gap // interval)
        for i in range(1, n + 1):
            t = a["t"] + gap * i / (n + 1)
            filled.append({"t": t, "kind": "fill", "score": 0.0})
    # Dedupe frames closer than 1s.
    frames = []
    for c in filled:
        if not frames or c["t"] - frames[-1]["t"] >= 1.0:
            frames.append(c)
        elif c["score"] > frames[-1]["score"]:
            frames[-1] = c
    if len(frames) > max_frames:
        # Keep coverage: one frame per equal time bucket, preferring the strongest scene cut.
        buckets: dict[int, dict] = {}
        for c in frames:
            k = min(int(c["t"] / max(duration, 1e-6) * max_frames), max_frames - 1)
            if k not in buckets or c["score"] > buckets[k]["score"]:
                buckets[k] = c
        frames = [buckets[k] for k in sorted(buckets)]
    return frames


def extract_frames(video: Path, frames: list[dict], out: Path) -> None:
    fdir = out / "frames"
    fdir.mkdir(exist_ok=True)
    scale = f"scale={FRAME_MAX_EDGE}:{FRAME_MAX_EDGE}:force_original_aspect_ratio=decrease"

    def grab(i_f):
        i, f = i_f
        name = f"f{i:03d}_{ts(f['t']).replace(':', '-')}.jpg"
        p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{f['t']:.2f}",
                 "-i", str(video), "-frames:v", "1", "-vf", scale, "-q:v", "3", str(fdir / name)])
        f["file"] = f"frames/{name}" if p.returncode == 0 and (fdir / name).exists() else None

    log(f"extracting {len(frames)} frames...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(grab, enumerate(frames, 1)))
    frames[:] = [f for f in frames if f.get("file")]


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


def transcribe_local(video: Path, out: Path, lang: str | None) -> list[dict]:
    audio = out / "audio.mp3"
    p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
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
    return segs


# ---------------------------------------------------------------- output

def write_outputs(out: Path, meta: dict, frames: list[dict], sheets: list[dict], segs: list[dict]) -> None:
    (out / "transcript.md").write_text(
        "\n".join(f"[{ts(s['start'])}] {s['text']}" for s in segs) + "\n", encoding="utf-8")

    L = [f"# {meta.get('title') or 'Untitled'}", ""]
    for k in ("source", "uploader", "upload_date", "duration", "transcript_source"):
        if meta.get(k):
            L.append(f"- **{k}**: {meta[k]}")
    L.append(f"- **frames**: {len(frames)} (scene cuts + gap fill, max edge {FRAME_MAX_EDGE}px)")
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


LOCAL_COPY_EXT = {"aac": "m4a", "alac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg", "flac": "flac"}


def extract_audio(source: str, is_local: bool, info: dict, vid: str, args) -> None:
    """Save the audio track (no transcription) for other tools; print its path on stdout."""
    out = Path(args.audio_dir).expanduser() if args.audio_dir else CACHE_ROOT / vid / "audio"
    out.mkdir(parents=True, exist_ok=True)
    fmt = args.audio_format
    if is_local:
        src = Path(source).expanduser()
        p = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                 "-of", "csv=p=0", str(src)])
        codec = p.stdout.strip()
        if not codec:
            sys.exit("no audio track in this file")
        ext = LOCAL_COPY_EXT.get(codec, "mka") if fmt == "original" else fmt
        dest = out / f"{vid}.{ext}"
        enc = ["-c:a", "copy"] if fmt == "original" else []
        p = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map", "a:0",
                 *enc, *args.ffmpeg_audio_opts, str(dest)])
        if p.returncode != 0:
            sys.exit(f"audio extraction failed: {p.stderr.strip()[-300:]}")
    else:
        cmd = ytdlp_base(args) + ["-f", "ba/b", "-x", "--audio-format", "best" if fmt == "original" else fmt,
                                  "-o", str(out / f"{vid}.%(ext)s")]
        if args.ffmpeg_audio_opts:
            cmd += ["--postprocessor-args", "ExtractAudio:" + " ".join(args.ffmpeg_audio_opts)]
        log("downloading audio...")
        p = run(cmd + ["--", source])
        files = [f for f in out.glob(f"{vid}.*") if f.suffix not in (".part", ".ytdl", ".json")]
        if p.returncode != 0 or not files:
            sys.exit(f"audio download failed: {p.stderr.strip()[-500:]}")
        dest = max(files, key=lambda f: f.stat().st_mtime)
    sidecar = {"source": source, "audio": str(dest), **media_info(dest)}
    if info:
        sidecar.update({k: info.get(k) for k in ("id", "title", "uploader", "upload_date", "description", "webpage_url")})
    dest.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"audio saved ({sidecar['duration']:.1f}s)")
    print(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video URL or local file path")
    ap.add_argument("--probe", action="store_true", help="print metadata only, no download")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--interval", type=float, default=5.0, help="max seconds between frames before capping")
    ap.add_argument("--no-frames", action="store_true", help="transcript only")
    ap.add_argument("--force-asr", action="store_true", help="ignore subtitles, transcribe audio locally")
    ap.add_argument("--lang", help="subtitle / ASR language hint, e.g. zh, en, ja")
    ap.add_argument("--cookies-from-browser", help="e.g. chrome, safari (for login-only videos)")
    ap.add_argument("--keep-video", action="store_true")
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

    for tool in ("ffmpeg", "ffprobe", "uv", YTDLP[0]):
        if not shutil.which(tool):
            sys.exit(f"missing dependency: {tool} (brew install {tool})")

    local = Path(args.source).expanduser()
    is_local = local.exists()
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

    variant = (("noframes" if args.no_frames else f"f{args.max_frames}i{args.interval:g}")
               + ("-asr" if args.force_asr else "") + (f"-{args.lang}" if args.lang else ""))
    out = CACHE_ROOT / vid / variant
    has_video = any(out.glob("video.*"))
    if (out / "timeline.md").exists() and not args.refresh and (has_video or not args.keep_video or is_local):
        log("cache hit")
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

    need_media = not args.no_frames or not segs
    video = None
    if need_media:
        video = local if is_local else download_video(args.source, out, args)
        mi = media_info(video)
        meta["duration"] = ts(mi["duration"])
        if not segs:
            if mi["has_audio"]:
                segs = transcribe_local(video, out, args.lang)
                if segs:
                    meta["transcript_source"] = "local Whisper ASR (mlx-whisper large-v3-turbo; may contain recognition errors)"
                else:
                    meta["limitations"].append("speech recognition failed; answer from frames only")
            else:
                meta["limitations"].append("video has no audio track")
    else:
        meta["duration"] = ts(info.get("duration") or 0)

    frames, sheets = [], []
    if not args.no_frames and video is not None:
        mi = media_info(video)
        if mi["has_video"]:
            frames = plan_frames(mi["duration"], detect_scenes(video), args.interval, args.max_frames)
            extract_frames(video, frames, out)
            sheets = make_sheets(frames, out)
            if mi["duration"] / max(len(frames), 1) > args.interval * 1.5:
                meta["limitations"].append(
                    f"long video: ~1 frame per {mi['duration'] / max(len(frames), 1):.0f}s on average")

    if video is not None and not is_local and not args.keep_video:
        video.unlink(missing_ok=True)

    write_outputs(out, meta, frames, sheets, segs)
    log(f"done: {len(frames)} frames, {len(sheets)} sheets, {len(segs)} transcript segments")
    print(out / "timeline.md")


if __name__ == "__main__":
    main()
