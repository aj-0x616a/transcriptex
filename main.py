
#!/usr/bin/env python3
"""
Transcriptex (Click-based CLI)

Features:
- Resolve YouTube video ID & title with yt-dlp
- Fetch transcript via youtube-transcript-api (OO interface: YouTubeTranscriptApi().fetch)
- Skip re-download if transcript already exists (use --force to override)
- Optionally generate headings with a local Ollama model via /api/chat using structured outputs
  (per-paragraph calls for reliable alignment)
- Write TXT + LaTeX outputs under ~/transcriptex (configurable with --base-dir)

Usage:
  transcriptex [OPTIONS] ID_OR_URL

Examples:
  transcriptex "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  transcriptex dQw4w9WgXcQ
  transcriptex dQw4w9WgXcQ --ollama-model llama3.2:3b
  transcriptex dQw4w9WgXcQ --force --paragraph-chars 600

Dependencies:
  pip install click yt-dlp youtube-transcript-api requests
"""

import os
import re
import sys
import json
from typing import List, Tuple, Optional

import click

# External deps (install with: pip install yt-dlp youtube-transcript-api)
try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

try:
    # Object-oriented interface
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

# requests for Ollama HTTP calls (install with: pip install requests)
try:
    import requests
except ImportError:
    requests = None


# ------------------------ Helpers ------------------------

def ensure_deps(require_ollama: bool = False):
    """Ensure mandatory deps are available. If Ollama is requested, ensure requests is present."""
    if YoutubeDL is None:
        raise SystemExit("Missing dependency: yt-dlp. Install with: pip install yt-dlp")
    if YouTubeTranscriptApi is None:
        raise SystemExit("Missing dependency: youtube-transcript-api. Install with: pip install youtube-transcript-api")
    if require_ollama and requests is None:
        raise SystemExit("Missing dependency: requests (needed for Ollama). Install with: pip install requests")


def expand_home_path(path: str) -> str:
    return os.path.expanduser(path)


def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    """
    Ensure <base_dir>/transcripts and <base_dir>/latex exist.
    Returns (transcripts_dir, latex_dir).
    """
    transcripts_dir = expand_home_path(os.path.join(base_dir, "transcripts"))
    latex_dir = expand_home_path(os.path.join(base_dir, "latex"))
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    return transcripts_dir, latex_dir


def slugify_filename(name: str, max_len: int = 120) -> str:
    """
    Make a safe filename from a video title: remove/replace problematic characters,
    collapse whitespace, and limit length.
    """
    name = re.sub(r'[\/\\:\*\?"<>\|]', "-", name)  # reserved/path chars
    name = re.sub(r"\s+", " ", name).strip()       # normalize whitespace
    # Keep simple ASCII in filenames to be safe across platforms
    name = "".join(ch if 32 <= ord(ch) < 127 else "-" for ch in name)
    name = name[:max_len].rstrip(" .")             # strip trailing dot/space
    return name or "untitled"


def escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters in plain text.
    """
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
    return "".join(specials.get(ch, ch) for ch in text)


def wrap_paragraphs(text_chunks: List[str], approx_chars: int = 800) -> List[str]:
    """
    Combine transcript text chunks into paragraphs of approximately `approx_chars`.
    Break on sentence boundaries when possible.
    """
    full = " ".join(t.strip() for t in text_chunks if t and t.strip())
    full = re.sub(r"\s+", " ", full).strip()

    if not full:
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
    return paragraphs


def build_latex_document(title: str, paragraphs: List[str], headings: Optional[List[str]] = None) -> str:
    """
    Create a minimal LaTeX document with a main heading and optional subheadings above each paragraph.
    Uses an rf-string with doubled braces to avoid f-string brace collisions.
    """
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

    return rf"""\documentclass[12pt]{{article}}
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


# ------------------------ YouTube functions ------------------------

def get_youtube_metadata(id_or_url: str) -> Tuple[str, str]:
    """
    Use yt-dlp to get canonical video ID and title from an ID or URL.
    Returns (video_id, title).
    """
    opts = {"quiet": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(id_or_url, download=False)
        video_id = info.get("id")
        title = info.get("title") or f"YouTube Video {video_id or 'unknown'}"
        if not video_id:
            raise RuntimeError("yt-dlp could not determine the video ID.")
        return video_id, title


def get_youtube_transcript(video_id: str) -> List[str]:
    """
    Fetch transcript via the OO interface:
      ytt_api = YouTubeTranscriptApi()
      transcript = ytt_api.fetch(video_id)

    The returned object is iterable/indexable; each snippet typically provides:
      - .text
      - .start
      - .duration

    Returns a list of snippet texts.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)

        texts: List[str] = []
        for snippet in fetched_transcript:
            text = getattr(snippet, "text", None)
            if text is None and isinstance(snippet, dict):
                text = snippet.get("text")
            if text:
                texts.append(text)
        return texts

    except VideoUnavailable:
        raise RuntimeError("The specified video is unavailable or private.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for the specified video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")


# ------------------------ Ollama integration (per-paragraph) ------------------------

def generate_heading_for_paragraph_ollama(
    paragraph: str,
    model: str = "llama3.2:3b",
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 30,
) -> Optional[str]:
    """
    Ask Ollama (via /api/chat) for ONE heading for ONE paragraph using structured outputs.
    Returns the heading string, or None on failure.
    """
    if requests is None:
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
        {
            "role": "user",
            "content": paragraph,
        },
    ]

    # JSON schema for a single string
    schema = {"type": "string"}

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,           # Structured output: a string
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 64,
        },
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        if not content:
            return None

        # content should be a JSON string due to 'format': {"type":"string"}
        try:
            heading = json.loads(content)
        except json.JSONDecodeError:
            # Some models might return plain text. Accept as-is.
            heading = content.strip()

        # sanitize
        heading = " ".join(str(heading).split()).rstrip(".!?:;")

        # minimal validity check
        if len(heading) == 0:
            return None

        return heading

    except Exception:
        return None


def generate_headings_with_ollama_per_paragraph(
    paragraphs: List[str],
    model: str,
    endpoint: str,
    timeout: int,
) -> List[str]:
    """
    Generate headings by calling Ollama once per paragraph.
    Returns a list of headings aligned to 'paragraphs'.
    """
    headings: List[str] = []
    for i, p in enumerate(paragraphs):
        h = generate_heading_for_paragraph_ollama(
            paragraph=p, model=model, endpoint=endpoint, timeout=timeout
        )
        if h is None:
            h = f"Section {i+1}"  # fallback only for this paragraph
        headings.append(h)
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
    help="Base directory for outputs (transcripts/ and latex/ will be created).",
)
@click.version_option(version="0.1.0", prog_name="transcriptex")
def transcriptex(id: str, service: str, paragraph_chars: int, force: bool,
                 ollama_model: Optional[str], ollama_endpoint: str, ollama_timeout: int,
                 base_dir: str):
    """
    Download transcript and produce LaTeX + TXT files under BASE_DIR (default: ~/transcriptex).
    If --ollama-model is provided, headings are generated via Ollama per paragraph; paragraph text remains unchanged.
    """
    require_ollama = bool(ollama_model)
    ensure_deps(require_ollama=require_ollama)

    if service.lower() != "youtube":
        click.echo("Only 'youtube' is supported at the moment.", err=True)
        sys.exit(2)

    # Resolve metadata (ID + title) using yt-dlp
    try:
        video_id, title = get_youtube_metadata(id)
    except Exception as e:
        click.echo(f"Error getting video metadata via yt-dlp: {e}", err=True)
        sys.exit(2)

    # Prepare output directories under base_dir
    base_dir_expanded = expand_home_path(base_dir)
    transcripts_dir, latex_dir = ensure_dirs(base_dir_expanded)

    # Construct safe filenames: <title>-<video_id>.<ext>
    base_name = f"{slugify_filename(title)}-{video_id}"
    transcript_path = os.path.join(transcripts_dir, f"{base_name}.txt")
    latex_path = os.path.join(latex_dir, f"{base_name}.tex")

    # If transcript exists and not forcing re-download, reuse it
    paragraphs: List[str] = []
    if os.path.exists(transcript_path) and not force:
        click.echo(f"🟡 Transcript already exists, skipping download:\n  {transcript_path}")

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()

        if transcript_text:
            if "\n\n" in transcript_text:
                paragraphs = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
            else:
                paragraphs = wrap_paragraphs([transcript_text], approx_chars=paragraph_chars)
        else:
            paragraphs = []
    else:
        # Fresh fetch
        try:
            chunks = get_youtube_transcript(video_id)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(3)

        paragraphs = wrap_paragraphs(chunks, approx_chars=paragraph_chars)
        transcript_text = "\n\n".join(paragraphs) if paragraphs else ""
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        click.echo(f"📄 Transcript written:\n  {transcript_path}")

    # Optionally generate headings via Ollama (per paragraph)
    headings: Optional[List[str]] = None
    if ollama_model:
        ensure_deps(require_ollama=True)
        click.echo(f"🤖 Generating headings with Ollama model '{ollama_model}' (per paragraph)...")
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

    click.echo("✅ Done.")
    click.echo(f"Transcript (TXT): {transcript_path}")
    click.echo(f"LaTeX (TEX):      {latex_path}")


if __name__ == "__main__":
    transcriptex()
