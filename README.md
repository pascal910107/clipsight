# clipsight

讓 Claude Code 看影片。

丟一個影片連結（YouTube、Instagram、TikTok 等 yt-dlp 支援的網站）或本機影片檔，它會抽出帶時間標記的畫面和逐字稿，交給 Claude 回答「這支影片在講什麼」、「第 3 分鐘那個畫面是什麼」之類的問題。

- 有字幕就用字幕，沒有才在本機用 Whisper 轉文字
- 依場景切換抽畫面，另外產生 3×3 縮圖總覽，省 token
- 也可以只抽音檔，給其他程式用

## 需求

- macOS（Apple Silicon）
- [uv](https://docs.astral.sh/uv/)、ffmpeg

```bash
brew install uv ffmpeg
```

其他依賴（yt-dlp、Pillow、mlx-whisper）第一次執行時由 uv 自動安裝。

## 安裝成 Claude Code skill

```bash
git clone https://github.com/pascal910107/clipsight ~/.claude/skills/clipsight
```

之後在 Claude Code 貼上影片連結問問題就會用到。

## 直接用

```bash
# 看影片資訊，不下載
./video_extract.py --probe <url>

# 抽畫面＋逐字稿，輸出 timeline.md 的路徑
./video_extract.py <url 或檔案>

# 只要逐字稿
./video_extract.py --no-frames <url>

# 只抽音檔（輸出音檔路徑）
./video_extract.py --audio-only <url>
./video_extract.py --audio-only --audio-format wav --sample-rate 16000 --mono <url>
```

需要登入才能看的影片，加上 `--cookies-from-browser chrome`。

結果快取在 `~/.cache/clipsight/`。

## 限制

- 看到的是抽樣畫面，不是連續影像，背景音樂和音效也聽不到
- 自動字幕和語音辨識可能有錯字
- 長影片最多 100 張畫面，越長越稀疏
