---
name: clipsight
description: Watch and analyze a video's actual content (visuals + speech) from a YouTube / Instagram / TikTok / any yt-dlp URL or a local video file, or save just its audio for other tools. Use when the user shares a video link or file and asks what it says or shows, wants a summary, notes, or a question answered about a specific moment ("這影片在說什麼", "幫我看這支影片", "第 3 分鐘畫面是什麼", "summarize this video"), or wants the audio extracted.
---

# clipsight

Turns a video into timestamped frames plus a transcript, then you look at the frames and read the transcript to answer. The video itself is never sent to the model.

Scripts live in this skill's base directory (written `<skill dir>` below). They need `ffmpeg` and `uv`.

## Steps

1. **Probe first** (fast, no download):
   `<skill dir>/video_extract.py --probe "<URL>"`
   You get the title, duration, approximate size, and which subtitle track will be used.
   - Over **60 minutes**: tell the user the duration and ask whether to process the whole thing, a time range (`--start/--end`), or suggest `--no-frames` (transcript only, very fast when subtitles exist). `chapters` in the probe output help pick a range.
   - No subtitles (`chosen_subtitle: null`) means local Whisper ASR, roughly 1/10 to 1/20 of real time on Apple Silicon. For long videos, mention that it will take a while.
   - A playlist URL is rejected. Ask the user which video they mean.

2. **Extract**:
   `<skill dir>/video_extract.py "<URL or path>"`
   It prints the path to `timeline.md`. Results are cached in `~/.cache/clipsight/<id>/<variant>/`, so rerunning is instant.
   Options:
   - `--start 10:00 --end 15:00`: analyze only this range; all 100 frames go to it (the whole video is still downloaded)
   - `--no-frames`: transcript only
   - `--max-frames N`: default 100
   - `--interval S`: target max gap between frames, default 5
   - `--no-dedup`: keep near-identical frames (by default static stretches collapse to one frame per minute at most)
   - `--force-asr`: ignore subtitles and transcribe the audio
   - `--lang zh|en|ja`: language hint for subtitles/ASR
   - `--keep-video`: keep the downloaded file
   - `--cookies-from-browser chrome`: for login, members-only, or age-gated videos; use only if the user agrees, since it uses their browser login
   - `--refresh`: ignore the cache

   For long runs, use `run_in_background`.

3. **Read `timeline.md`**: metadata, chapters, description, a list of contact sheets, then a timeline where each frame is followed by the speech until the next frame.

4. **Look at the visuals**, matching depth to the question. Images cost tokens: a contact sheet is about 1.6k, a single frame about 450.
   - Overview or summary: Read the contact sheets in `sheets/`. Each holds 3×3 frames with a yellow timestamp. For speech-heavy videos (talking head, podcast), 2–3 sheets are usually enough.
   - Specific moment, small text, code, slides, charts: Read the individual `frames/fNNN_HH-MM-SS.jpg` (768px) near that time.
   - Need a moment between sampled frames, or sharper detail (up to 1280px)?
     `<skill dir>/video_extract.py "<URL>" --at 03:12,05:40` prints one image path per time. The video stays cached, so later `--at` calls are fast.

5. **Answer** in the user's language, citing timestamps `[mm:ss]` so they can jump there. For a YouTube URL, you can give a link like `https://youtu.be/<id>?t=<seconds>`.

## Audio only

`<skill dir>/video_extract.py --audio-only "<URL or path>"` prints the saved audio file path on stdout. Next to it is a `.json` sidecar with title, uploader, and duration.
- `--audio-format original` (default): no re-encode, best quality. Also `wav`, `flac`, `mp3`, `m4a`.
- `--sample-rate 16000 --mono`: resample and downmix; needs a re-encoded format.
- `--audio-dir DIR`: output folder (default `~/.cache/clipsight/<id>/audio/`).
- `--start/--end`: cut a range.

Instagram often requires login: if it fails with "login required" or "rate-limit", retry with `--cookies-from-browser chrome` after the user agrees.

## Limits (tell the user when relevant)

- Frames are samples. Motion, fast action between frames, and music/sound effects are not perceived.
- Auto-captions and Whisper can mishear names, terms, and sung lyrics. `timeline.md` lists `transcript_source`.
- If `limitations` mentions no audio or failed ASR, answer from the frames only and say so.
- Long videos are capped at 100 frames, so frames are sparse (the limitations line gives the average gap). For a specific segment, rerun with `--start/--end` or use `--at`.
- Local transcription uses mlx-whisper, which only runs on Apple Silicon Macs. Elsewhere, only videos with subtitles get a transcript.
