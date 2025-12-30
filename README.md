# **Transcriptex**

Transcriptex is a Python CLI tool that:

✅ Downloads transcripts from YouTube videos  
✅ Converts them into **LaTeX** documents for offline reading  
✅ Supports **local Whisper transcription** or **OpenAI API transcription**  
✅ Optionally generates **headings for each paragraph** using a local **Ollama model** (runs entirely on your machine for privacy)  

---

## **Features**
- Resolve YouTube video ID & title using `yt-dlp`.
- Fetch transcript via `youtube-transcript-api` or transcribe audio using:
  - **Local Whisper** (`openai-whisper`) – default mode.
  - **OpenAI API** (`gpt-4o-transcribe`) – optional.
- Skip work if transcript already exists (`--force` to override).
- Generate semantic headings per paragraph using **Ollama** with structured outputs.
- Save outputs as:
  - **Plain text** (`~/transcriptex/transcripts`)
  - **LaTeX** (`~/transcriptex/latex`)
- Reuse existing audio files in `~/transcriptex/audio` when available.
- Logging with configurable level and optional log file.

---

## **Requirements**
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) or pip for installation
- Dependencies:
```bash
click yt-dlp youtube-transcript-api requests openai openai-whisper
```
- Optional: [Ollama](https://ollama.ai) running locally for headings.

---

## **Installation**
### **Using uv (recommended)**
```bash
uv tool install "git+https://github.com/aj-0x616a/transcriptex.git"
```

### **Using pip**
```bash
pip install git+https://github.com/aj-0x616a/transcriptex.git
```

---

## **Usage**
### **Basic**
```bash
transcriptex "<youtube-url-or-id>"
```
Example:
```bash
transcriptex "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### **Choose transcription mode**
- Local Whisper (default):
```bash
transcriptex "<youtube-url-or-id>" --whisper-mode local
```
- OpenAI API:
```bash
transcriptex "<youtube-url-or-id>" --whisper-mode api --whisper-model gpt-4o-transcribe
```

### **Generate headings with Ollama**
```bash
transcriptex "<youtube-url-or-id>" --ollama-model llama3.2:3b
```

### **Other options**
- Force re-download:
```bash
transcriptex "<youtube-url-or-id>" --force
```
- Change paragraph size:
```bash
transcriptex "<youtube-url-or-id>" --paragraph-chars 600
```
- Change output directory:
```bash
transcriptex "<youtube-url-or-id>" --base-dir ~/my-transcripts
```
- Custom Ollama endpoint or timeout:
```bash
transcriptex "<youtube-url-or-id>" --ollama-endpoint http://localhost:11434/api/chat --ollama-timeout 45
```
- Keep audio files:
```bash
transcriptex "<youtube-url-or-id>" --keep-audio
```
- Logging options:
```bash
transcriptex "<youtube-url-or-id>" --log-level DEBUG --log-file ./transcriptex.log
```

---

## **Example Output**
- Transcript saved to:
```
~/transcriptex/transcripts/<video-title>-<video-id>.txt
```
- LaTeX saved to:
```
~/transcriptex/latex/<video-title>-<video-id>.tex
```

---

## **Why Ollama?**
- Runs locally for **privacy** and **speed**.
- Structured outputs ensure clean headings without modifying paragraph content.
- Recommended model: `llama3.2:3b` for balance of speed and quality.

---

## **License**
Apache License 2.0
