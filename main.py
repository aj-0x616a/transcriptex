"""
Transcriptex: Download a YouTube transcript (OO interface) via youtube-transcript-api,
get the video title via yt-dlp, and write both a minimal LaTeX file and a plain-text
transcript to your home directory.

Outputs:
  ~/transcriptex/transcripts/<title>-<video_id>.txt
  ~/transcriptex/latex/<title>-<video_id>.tex

Usage examples:
  python transcriptex.py --id https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python transcriptex.py --id dQw4w9WgXcQ
  python transcriptex.py --service youtube --id dQw4w9WgXcQ
"""

import argparse
import os
import re
import sys
from typing import List, Tuple

# External deps:
#   pip install yt-dlp youtube-transcript-api
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


# ------------------------ CLI args ------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcriptex",
        description="Download a YouTube transcript and output LaTeX + TXT files."
    )
    parser.add_argument(
        "--service",
        default="youtube",
        choices=["youtube"],
        help="Transcript source service (default: youtube)."
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Video ID or URL (for YouTube)."
    )
    parser.add_argument(
        "--paragraph-chars",
        type=int,
        default=800,
        help="Approx. characters per paragraph when wrapping transcript text (default: 800)."
    )
    return parser.parse_args()


# ------------------------ Helpers ------------------------

def ensure_deps():
    if YoutubeDL is None:
        raise SystemExit("Missing dependency: yt-dlp. Install with: pip install yt-dlp")
    if YouTubeTranscriptApi is None:
        raise SystemExit("Missing dependency: youtube-transcript-api. Install with: pip install youtube-transcript-api")


def expand_home_path(path: str) -> str:
    return os.path.expanduser(path)


def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    """
    Ensure ~/transcriptex/transcripts and ~/transcriptex/latex exist.
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
    # Replace path separators and reserved characters
    name = re.sub(r'[\/\\:\*\?"<>\|]', "-", name)
    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Replace remaining non-ASCII with hyphen (simple guard)
    name = "".join(ch if 32 <= ord(ch) < 127 else "-" for ch in name)
    # Trim and strip trailing dot/space
    name = name[:max_len].rstrip(" .")
    # Fallback if empty
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
    Breaks on sentence boundaries when possible.
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


def build_latex_document(title: str, paragraphs: List[str]) -> str:
    """
    Create a minimal LaTeX document with a main heading and paragraphs.
    Use rf-string with doubled braces to avoid f-string/format placeholder collisions.
    """
    escaped_title = escape_latex(title)
    body = "\n\n".join(escape_latex(p) for p in paragraphs)

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
    Fetch the transcript via the OO interface:
      ytt_api = YouTubeTranscriptApi()
      transcript = ytt_api.fetch(video_id)

    The returned object is iterable/indexable and each snippet provides:
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


# ------------------------ Main ------------------------

def main():
    args = parse_args()
    ensure_deps()

    if args.service != "youtube":
        print("Only 'youtube' is supported at the moment.", file=sys.stderr)
        sys.exit(2)

    # Resolve metadata (ID + title) using yt-dlp
    try:
        video_id, title = get_youtube_metadata(args.id)
    except Exception as e:
        print(f"Error getting video metadata via yt-dlp: {e}", file=sys.stderr)
        sys.exit(2)

    # Fetch transcript via OO interface
    try:
        chunks = get_youtube_transcript(video_id)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)

    paragraphs = wrap_paragraphs(chunks, approx_chars=args.paragraph_chars)

    # Build outputs
    latex_doc = build_latex_document(title, paragraphs)
    transcript_text = "\n\n".join(paragraphs) if paragraphs else ""

    # Prepare output directories under ~/transcriptex
    base_dir = expand_home_path("~/transcriptex")
    transcripts_dir, latex_dir = ensure_dirs(base_dir)

    # Construct safe filenames: <title>-<video_id>.<ext>
    base_name = f"{slugify_filename(title)}-{video_id}"
    transcript_path = os.path.join(transcripts_dir, f"{base_name}.txt")
    latex_path = os.path.join(latex_dir, f"{base_name}.tex")

    # Write files
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_doc)

    print("✅ Done.")
    print(f"Transcript (TXT): {transcript_path}")
    print(f"LaTeX (TEX):      {latex_path}")


if __name__ == "__main__":
    main()
