# Transcriptex

Transcriptex is a Python CLI tool that:

✅ Downloads transcripts from YouTube videos  
✅ Converts them into **LaTeX** documents for offline reading  
✅ Optionally generates **headings for each paragraph** using a local **Ollama model** (runs entirely on your machine for privacy)  

---

## **Features**
- Fetch YouTube video title and transcript using `yt-dlp` and `youtube-transcript-api`.
- Save outputs as:
  - **Plain text** (`~/transcriptex/transcripts`)
  - **LaTeX** (`~/transcriptex/latex`)
- Skip re-download if transcript already exists (`--force` to override).
- Generate semantic headings per paragraph using **Ollama** with structured outputs.
- Fully local workflow (no external API calls).

---

## **Requirements**
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) or pip for installation
- Dependencies:
  ```bash
  click yt-dlp youtube-transcript-api requests
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
