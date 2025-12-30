
#!/usr/bin/env python3
"""
Transcriptex (Click-based CLI)

Features:
- Resolve YouTube video ID & title with yt-dlp (with external JS runtime support)
- Fetch transcript via youtube-transcript-api (class methods)
- Local Whisper transcription (default) using open-source `openai-whisper`
  * Optional API mode with OpenAI Speech-to-Text (off by default)
- Skips work if transcript already exists (use --force to override)
- Optional per-paragraph headings with a local Ollama model via /api/chat (structured outputs)
- Write TXT + LaTeX outputs under BASE_DIR (default: ~/transcriptex)
- Reuse existing downloaded audio in BASE_DIR/audio/<base_name>.<ext> when present
- Logging via Python `logging` (use --log-level, --log-file)
- yt-dlp JS runtime auto-detection (Deno/Node/Bun/QuickJS) or --js-runtime override

Usage:
  transcriptex [OPTIONS] ID_OR_URL
"""

import os
import re
import sys
import json
import logging
import shutil
from typing import List, Tuple, Optional, Dict

import click

# Global logger
logger = logging.getLogger("transcriptex")

# ------------------------ External deps ------------------------
try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

try:
    # Class-method interface
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception

# requests for Ollama HTTP calls
try:
    import requests
except ImportError:
        requests = None

# OpenAI API (optional; used only if whisper-mode=api)
try:
    from openai import OpenAI  # OpenAI Python SDK ≥ 1.x
except ImportError:
    OpenAI = None

# Local Whisper (open-source; default)
try:
    import whisper as whisper_local  # pip/uv install openai-whisper
except ImportError:
    whisper_local = None


# ------------------------ Helpers ------------------------

def ensure_deps(require_ollama: bool = False,
                require_openai_api: bool = False,
                require_local_whisper: bool = False):
    """Ensure mandatory deps; optionally ensure ollama/openai-whisper/openai SDK."""
    if YoutubeDL is None:
        logger.error("Missing dependency: yt-dlp. Install with: pip/uv install yt-dlp")
        raise SystemExit(1)
    if YouTubeTranscriptApi is None:
        logger.error("Missing dependency: youtube-transcript-api. Install with: pip/uv install youtube-transcript-api")
        raise SystemExit(1)
    if require_ollama and requests is None:
        logger.error("Missing dependency: requests (needed for Ollama). Install with: pip/uv install requests")
        raise SystemExit(1)
    if require_openai_api and OpenAI is None:
        logger.error("Missing dependency: openai (SDK). Install with: pip/uv install openai")
        raise SystemExit(1)
    if require_local_whisper and whisper_local is None:
        logger.error("Missing dependency: openai-whisper. Install with: pip/uv install openai-whisper")
        raise SystemExit(1)


def expand_home_path(path: str) -> str:
    return os.path.expanduser(path)


def ensure_dirs(base_dir: str) -> Tuple[str, str, str]:
    """
    Ensure <base_dir>/transcripts, <base_dir>/latex, and <base_dir>/audio exist.
    Returns (transcripts_dir, latex_dir, audio_dir).
    """
    transcripts_dir = expand_home_path(os.path.join(base_dir, "transcripts"))
    latex_dir = expand_home_path(os.path.join(base_dir, "latex"))
    audio_dir = expand_home_path(os.path.join(base_dir, "audio"))
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    logger.debug("Ensured directories: transcripts=%s latex=%s audio=%s",
                 transcripts_dir, latex_dir, audio_dir)
    return transcripts_dir, latex_dir, audio_dir


def slugify_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[\/\\:\*\?"<>|]', "-", name)  # reserved/path chars
    name = re.sub(r"\s+", " ", name).strip()       # normalize whitespace
    name = "".join(ch if 32 <= ord(ch) < 127 else "-" for ch in name)
    name = name[:max_len].rstrip(" .")             # strip trailing dot/space
    result = name or "untitled"
    logger.debug("Slugified filename: input=%r output=%r", name, result)
    return result


def escape_latex(text: str) -> str:
    specials = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "~": r"\textasciitilde{}",
        "_": r"\_",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(specials.get(ch, ch) for ch in text)
    return escaped


def wrap_paragraphs(text_chunks: List[str], approx_chars: int = 800) -> List[str]:
    full = " ".join(t.strip() for t in text_chunks if t and t.strip())
    full = re.sub(r"\s+", " ", full).strip()
    if not full:
        logger.info("No transcript text to wrap into paragraphs.")
        return []
    sentences = re.split(r"(?<=[\.!?])\s+", full)

    paragraphs, buf, buf_len = [], [], 0
    for s in sentences:
        buf.append(s)
        buf_len += len(s) + 1
        if buf_len >= approx_chars:
            paragraphs.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        paragraphs.append(" ".join(buf))
    logger.debug("Wrapped into %d paragraphs (approx_chars=%d)", len(paragraphs), approx_chars)
    return paragraphs


def build_latex_document(title: str, paragraphs: List[str], headings: Optional[List[str]] = None) -> str:
    escaped_title = escape_latex(title)
    blocks = []
    for i, p in enumerate(paragraphs):
        heading = None
        if headings and i < len(headings):
            heading = headings[i]
        if heading:
            blocks.append(rf"\subsection*{{{escape_latex(heading)}}}")
        blocks.append(escape_latex(p))

    body = "\n\n".join(blocks)

    latex = rf"""\documentclass[12pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage[margin=1in]{{geometry}}

\begin{{document}}

\section*{{Transcript: {escaped_title}}}

{body}

\end{{document}}
"""
    logger.debug("Built LaTeX document for title=%r, paragraphs=%d", title, len(paragraphs))
    return latex


# ------------------------ yt-dlp JS runtime selection ------------------------

def _parse_runtime_spec(spec: str) -> Dict[str, Dict[str, str]]:
    """
    Convert 'runtime' or 'runtime:/path/to/bin' to {'runtime': {'path': '/path/to/bin'}}.
    If path not provided, try shutil.which(runtime); include empty dict if not found.
    """
    parts = spec.split(":", 1)
    runtime = parts[0].strip().lower()
    path = parts[1].strip() if len(parts) == 2 and parts[1].strip() else shutil.which(runtime)
    return {runtime: ({'path': path} if path else {})}

def build_js_runtimes_dict(override: Optional[str] = None) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Build a dict suitable for yt-dlp's 'js_runtimes' Python API:
      {'deno': {'path': '/usr/bin/deno'}} or {'node': {'path': '/usr/local/bin/node'}}.
    Priority: override -> env(YTDLP_JS_RUNTIME) -> auto-detect [deno, node, bun, quickjs].
    If no runtime is found, return None (yt-dlp may enable deno by default if available).
    """
    if override:
        return _parse_runtime_spec(override)

    env_val = os.environ.get("YTDLP_JS_RUNTIME")
    if env_val:
        return _parse_runtime_spec(env_val)

    candidates = [
        ("deno", shutil.which("deno")),
        ("node", shutil.which("node")),
        ("bun", shutil.which("bun")),
        ("quickjs", shutil.which("qjs")),  # QuickJS commonly installed as 'qjs'
    ]
    for name, path in candidates:
        if path:
            return {name: {'path': path}}
    return None

def _log_js_runtime(js_rt_dict: Optional[Dict[str, Dict[str, str]]], context: str):
    """
    Log at INFO level which JS runtime is used (name + path), or that none was provided.
    """
    if js_rt_dict:
        # Typically a single runtime entry, but iterate defensively
        for name, cfg in js_rt_dict.items():
            path = cfg.get('path') or 'PATH lookup / default'
            logger.info("yt-dlp JS runtime (%s): %s (%s)", context, name, path)
    else:
        logger.info(
            "yt-dlp JS runtime (%s): none provided. yt-dlp will use its defaults "
            "(Deno enabled by default if available).", context
        )


# ------------------------ YouTube functions ------------------------

def get_youtube_metadata(id_or_url: str, js_runtime: Optional[str] = None) -> Tuple[str, str]:
    js_rt_dict = build_js_runtimes_dict(override=js_runtime)
    opts = {
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }
    if js_rt_dict:
        opts["js_runtimes"] = js_rt_dict

    _log_js_runtime(js_rt_dict, context="metadata")

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(id_or_url, download=False)
        video_id = info.get("id")
        title = info.get("title") or f"YouTube Video {video_id or 'unknown'}"
        if not video_id:
            raise RuntimeError("yt-dlp could not determine the video ID.")
        logger.info("Resolved YouTube metadata: id=%s title=%r", video_id, title)
        logger.debug(
            "Tip: External JS runtime improves format coverage; see yt-dlp EJS guide."
        )
        return video_id, title


def get_youtube_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[str]:
    """
    Return a list of text chunks for the video's transcript.
    If languages is provided, the API will try those first (e.g., ['en', 'en-US']).
    """
    try:
        logger.info("Fetching YouTube transcript for id=%s (languages=%s)", video_id, languages)
        items = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        chunks = [item.get("text", "") for item in items if item.get("text")]
        logger.info("Fetched %d transcript items from YouTube", len(chunks))
        return chunks
    except VideoUnavailable:
        raise RuntimeError("The specified video is unavailable or private.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for the specified video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")


# ------------------------ Audio download (yt-dlp) ------------------------

def download_audio_with_ytdlp(id_or_url: str, audio_dir: str, base_name: str,
                              js_runtime: Optional[str] = None) -> str:
    """
    Download best available audio-only stream with yt-dlp without re-encoding.
    Returns the local path to the downloaded file (.webm, .m4a, etc).
    """
    js_rt_dict = build_js_runtimes_dict(override=js_runtime)
    outtmpl = os.path.join(audio_dir, f"{base_name}.%(ext)s")
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }
    if js_rt_dict:
        opts["js_runtimes"] = js_rt_dict

    _log_js_runtime(js_rt_dict, context="download")

    logger.info("Downloading audio (bestaudio) to: %s", outtmpl)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(id_or_url, download=True)
        # Contextual tip: SABR-only formats may be missing direct URLs; yt-dlp picks available ones.
        logger.debug(
            "If some web/web_safari formats are missing URLs, YouTube may be forcing SABR streaming; "
            "yt-dlp will select available formats. See issue #12482."
        )
        path = ydl.prepare_filename(info)  # actual path with chosen extension
        logger.info("Downloaded audio file: %s", path)
        return path


def _find_existing_audio(audio_dir: str, base_name: str) -> Optional[str]:
    """
    Return the path to an existing audio file whose name starts with base_name and has an extension.
    If multiple exist, prefer the most recently modified.
    """
    if not os.path.isdir(audio_dir):
        logger.debug("Audio directory does not exist: %s", audio_dir)
        return None
    candidates: List[str] = []
    prefix = f"{base_name}."
    for fname in os.listdir(audio_dir):
        if fname.startswith(prefix):
            full = os.path.join(audio_dir, fname)
            if os.path.isfile(full):
                candidates.append(full)
    if not candidates:
        logger.debug("No existing audio found for base_name: %s", base_name)
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = candidates[0]
    logger.info("Found existing audio for reuse: %s", chosen)
    return chosen


def get_or_download_audio(id_or_url: str, audio_dir: str, base_name: str,
                          js_runtime: Optional[str] = None) -> Tuple[str, bool]:
    """
    Reuse existing audio file if present; otherwise download it.
    Returns (audio_path, downloaded_now_flag).
    """
    existing = _find_existing_audio(audio_dir, base_name)
    if existing:
        logger.info("Reusing existing audio file: %s", existing)
        return existing, False
    audio_path = download_audio_with_ytdlp(id_or_url, audio_dir, base_name, js_runtime=js_runtime)
    return audio_path, True


# ------------------------ Whisper transcription ------------------------

def transcribe_with_openai_api(audio_path: str, model: str = "gpt-4o-transcribe",
                               response_format: str = "text") -> str:
    """
    Transcribe audio via OpenAI Speech-to-Text API (optional).
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Install with: pip/uv install openai")

    logger.info("Transcribing with OpenAI API (model=%s)", model)
    client = OpenAI()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format=response_format,
        )
    text = getattr(resp, "text", str(resp))
    logger.info("Transcription (API) completed, length=%d chars", len(text or ""))
    return text


def transcribe_with_local_whisper(audio_path: str,
                                  model: str = "tiny.en",
                                  language: Optional[str] = None,
                                  task: str = "transcribe") -> str:
    """
    Transcribe audio locally using open-source Whisper (default path).
    """
    if whisper_local is None:
        raise RuntimeError("openai-whisper not installed. Install with: pip/uv install openai-whisper")

    logger.info("Transcribing locally with Whisper (model=%s, task=%s, language=%s)",
                model, task, language)
    wmodel = whisper_local.load_model(model)
    # fp16=False helps on CPU; if CUDA is available, Whisper uses it automatically.
    result = wmodel.transcribe(audio_path, language=language, task=task, fp16=False)
    text = result.get("text", "").strip()
    logger.info("Transcription (local) completed, length=%d chars", len(text or ""))
    return text


# ------------------------ Ollama integration (per-paragraph headings) ------------------------

def generate_heading_for_paragraph_ollama(
    paragraph: str,
    model: str = "llama3.2:3b",
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 30,
) -> Optional[str]:
    if requests is None:
        logger.warning("Requests not installed; skipping Ollama heading generation.")
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You generate a single heading for the given paragraph. "
                "Constraints: Do NOT rewrite or paraphrase the paragraph; "
                "return ONLY a single string (the heading). "
                "Style: Title Case, 5–9 words, no trailing punctuation."
            ),
        },
        {"role": "user", "content": paragraph},
    ]

    schema = {"type": "string"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 64},
    }

    try:
        logger.debug("Calling Ollama endpoint for heading (model=%s, timeout=%s)", model, timeout)
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            logger.warning("Ollama returned empty content for heading.")
            return None

        try:
            heading = json.loads(content)
        except json.JSONDecodeError:
            heading = content.strip()

        heading = " ".join(str(heading).split()).rstrip(".!?:;")
        logger.info("Generated heading: %s", heading)
        return heading or None
    except Exception as e:
        logger.warning("Ollama heading generation failed: %s", e)
        return None


def generate_headings_with_ollama_per_paragraph(
    paragraphs: List[str],
    model: str,
    endpoint: str,
    timeout: int,
) -> List[str]:
    headings: List[str] = []
    logger.info("Generating headings for %d paragraphs via Ollama", len(paragraphs))
    for i, p in enumerate(paragraphs):
        h = generate_heading_for_paragraph_ollama(
            paragraph=p, model=model, endpoint=endpoint, timeout=timeout
        )
        final = h or f"Section {i+1}"
        headings.append(final)
    logger.info("Generated %d headings.", len(headings))
    return headings


# ------------------------ Click CLI ------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("id", metavar="ID_OR_URL")
@click.option(
    "--service",
    type=click.Choice(["youtube"], case_sensitive=False),
    default="youtube",
    show_default=True,
    help="Transcript source service.",
)
@click.option(
    "--transcriber",
    type=click.Choice(["auto", "youtube", "whisper"], case_sensitive=False),
    default="whisper",
    show_default=True,
    help="Choose transcript source: auto (YouTube then Whisper fallback), youtube only, or whisper only.",
)
@click.option(
    "--whisper-mode",
    type=click.Choice(["api", "local"], case_sensitive=False),
    default="local",
    show_default=True,
    help="When transcriber=whisper or auto-fallback, choose API (OpenAI) or local (openai-whisper).",
)
@click.option(
    "--whisper-model",
    default=None,
    help="Model for Whisper transcription. Local: tiny.en/large-v3-(default tiny.en). API: gpt-4o-transcribe/whisper-1.",
)
@click.option(
    "--whisper-language",
    default=None,
    help="Optional ISO language code (e.g., 'en', 'es'). Helps accuracy for local Whisper.",
)
@click.option(
    "--whisper-task",
    type=click.Choice(["transcribe", "translate"], case_sensitive=False),
    default="transcribe",
    show_default=True,
    help="Local Whisper task: transcribe (same language) or translate (to English).",
)
@click.option(
    "--paragraph-chars",
    type=int,
    default=800,
    show_default=True,
    help="Approx. characters per paragraph when wrapping transcript text.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-download of transcript even if it already exists.",
)
@click.option(
    "--ollama-model",
    default=None,
    help="Use a local Ollama model (e.g., 'llama3.2:3b') to generate headings per paragraph.",
)
@click.option(
    "--ollama-endpoint",
    default="http://localhost:11434/api/chat",
    show_default=True,
    help="Ollama HTTP endpoint (use /api/chat for structured outputs).",
)
@click.option(
    "--ollama-timeout",
    type=int,
    default=60,
    show_default=True,
    help="Timeout for Ollama requests in seconds.",
)
@click.option(
    "--base-dir",
    default="~/transcriptex",
    show_default=True,
    help="Base directory for outputs (transcripts/, latex/, audio/ will be created).",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    default=False,
    help="Keep downloaded audio files (in BASE_DIR/audio).",
)
@click.option(
    "--js-runtime",
    default=None,
    help=("External JS runtime for yt-dlp (e.g., 'deno', 'node:/usr/local/bin/node'). "
          "If omitted, Transcriptex auto-detects or uses YTDLP_JS_RUNTIME env var."),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=True),
    default="INFO",
    show_default=True,
    help="Logging verbosity.",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True, allow_dash=True),
    default=None,
    help="Optional path to write logs to a file instead of stderr.",
)
@click.version_option(version="0.4.2", prog_name="transcriptex")
def transcriptex(id: str, service: str, transcriber: str, whisper_mode: str, whisper_model: Optional[str],
                 whisper_language: Optional[str], whisper_task: str,
                 paragraph_chars: int, force: bool,
                 ollama_model: Optional[str], ollama_endpoint: str, ollama_timeout: int,
                 base_dir: str, keep_audio: bool,
                 js_runtime: Optional[str],
                 log_level: str, log_file: Optional[str]):
    """
    Download transcript and produce LaTeX + TXT under BASE_DIR (default: ~/transcriptex).
    If --ollama-model is provided, headings are generated via Ollama per paragraph; paragraph text remains unchanged.
    """

    # ---------- Configure logging ----------
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = []
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file and log_file != "-":
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        handlers.append(fh)
    else:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        handlers.append(sh)

    logging.basicConfig(level=level, handlers=handlers)
    logger.info("Starting Transcriptex (log_level=%s, log_file=%s)", log_level, log_file or "stderr")

    # ---------- Dependency checks ----------
    require_ollama = bool(ollama_model)
    require_openai_api = (whisper_mode.lower() == "api" and transcriber.lower() in {"whisper", "auto"})
    require_local_whisper = (whisper_mode.lower() == "local" and transcriber.lower() in {"whisper", "auto"})
    ensure_deps(require_ollama=require_ollama,
                require_openai_api=require_openai_api,
                require_local_whisper=require_local_whisper)

    if service.lower() != "youtube":
        logger.error("Only 'youtube' is supported at the moment.")
        sys.exit(2)

    # Resolve metadata (ID + title) using yt-dlp
    try:
        video_id, title = get_youtube_metadata(id, js_runtime=js_runtime)
    except Exception as e:
        logger.error("Error getting video metadata via yt-dlp: %s", e)
        sys.exit(2)

    # Prepare output directories
    base_dir_expanded = expand_home_path(base_dir)
    transcripts_dir, latex_dir, audio_dir = ensure_dirs(base_dir_expanded)

    base_name = f"{slugify_filename(title)}-{video_id}"
    transcript_path = os.path.join(transcripts_dir, f"{base_name}.txt")
    latex_path = os.path.join(latex_dir, f"{base_name}.tex")

    # If transcript exists and not forcing re-download, reuse it
    paragraphs: List[str] = []
    reused = False
    if os.path.exists(transcript_path) and not force:
        logger.info("Transcript already exists; skipping download: %s", transcript_path)
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
        if transcript_text:
            if "\n\n" in transcript_text:
                paragraphs = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
            else:
                paragraphs = wrap_paragraphs([transcript_text], approx_chars=paragraph_chars)
        reused = True

    if not reused:
        # Decide path: YouTube transcript OR Whisper
        use_youtube = (transcriber.lower() == "youtube")
        use_whisper = (transcriber.lower() == "whisper")

        chunks: List[str] = []
        if use_youtube or (transcriber.lower() == "auto"):
            try:
                yt_langs = [whisper_language] if whisper_language else None
                chunks = get_youtube_transcript(video_id, languages=yt_langs)
            except RuntimeError as e:
                if transcriber.lower() == "youtube":
                    logger.error("YouTube transcript error: %s", e)
                    sys.exit(3)
                logger.warning("YouTube transcript unavailable; will try Whisper. Reason: %s", e)
                chunks = []

        if chunks:
            paragraphs = wrap_paragraphs(chunks, approx_chars=paragraph_chars)
            transcript_text = "\n\n".join(paragraphs) if paragraphs else ""
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            logger.info("Transcript written (YouTube): %s", transcript_path)
        else:
            # Whisper transcription path (default behavior)
            audio_file_path, downloaded_now = get_or_download_audio(id, audio_dir, base_name, js_runtime=js_runtime)
            file_size = os.path.getsize(audio_file_path)
            logger.info("Audio file size: %d bytes", file_size)

            # Choose Whisper mode & model
            if not whisper_model:
                whisper_model = "tiny.en" if whisper_mode.lower() == "local" else "gpt-4o-transcribe"

            transcript_text = ""
            try:
                if whisper_mode.lower() == "api":
                    # API upload has ~25MB file limit; if exceeded, switch to local
                    if file_size > 25 * 1024 * 1024:
                        logger.warning("Audio > 25MB; switching to local Whisper (open-source).")
                        transcript_text = transcribe_with_local_whisper(
                            audio_file_path, model="tiny.en", language=whisper_language, task=whisper_task
                        )
                    else:
                        transcript_text = transcribe_with_openai_api(
                            audio_file_path, model=whisper_model, response_format="text"
                        )
                else:
                    transcript_text = transcribe_with_local_whisper(
                        audio_file_path, model=whisper_model, language=whisper_language, task=whisper_task
                    )
            except Exception as e:
                logger.error("Error during Whisper transcription: %s", e)
                sys.exit(4)

            # Wrap to paragraphs
            paragraphs = wrap_paragraphs([transcript_text], approx_chars=paragraph_chars)

            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(paragraphs))
            logger.info("Transcript written (Whisper %s): %s", whisper_mode, transcript_path)

            # Cleanup audio unless requested (and only if we downloaded now)
            if not keep_audio and downloaded_now:
                try:
                    os.remove(audio_file_path)
                    logger.info("Removed audio file: %s", audio_file_path)
                except Exception as e:
                    logger.warning("Failed to remove audio file %s: %s", audio_file_path, e)

    # Optionally generate headings via Ollama (per paragraph)
    headings: Optional[List[str]] = None
    if ollama_model:
        ensure_deps(require_ollama=True)
        logger.info("Generating headings with Ollama model '%s' (per paragraph)...", ollama_model)
        headings = generate_headings_with_ollama_per_paragraph(
            paragraphs=paragraphs,
            model=ollama_model,
            endpoint=ollama_endpoint,
            timeout=ollama_timeout,
        )

    # Build LaTeX with (optional) headings
    latex_doc = build_latex_document(title, paragraphs, headings=headings)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_doc)
    logger.info("LaTeX written: %s", latex_path)

    logger.info("Done.")
    logger.info("Transcript (TXT): %s", transcript_path)
    logger.info("LaTeX (TEX):      %s", latex_path)


if __name__ == "__main__":
    transcriptex()
