"""YouTube transcript extraction."""

from __future__ import annotations

import re

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

_VIDEO_ID_PATTERNS = (
    r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/shorts\/)([0-9A-Za-z_-]{11})",
)


class YoutubeExtractionError(Exception):
    pass


def extract_video_id(url: str) -> str:
    for pattern in _VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise YoutubeExtractionError(f"Could not find a video id in URL: {url}")


def fetch_transcript(url: str) -> tuple[str, str]:
    """Return (transcript_text, display_name) for a YouTube video URL."""
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()
    try:
        # youtube-transcript-api >=1.0 replaced the old `YouTubeTranscriptApi.get_transcript(id)`
        # classmethod with an instance `.fetch(id)` call returning a FetchedTranscript of
        # FetchedTranscriptSnippet objects (`.text`, not the old dict's ["text"]). `.fetch()`
        # defaults to requesting English only - plenty of videos only have auto-generated
        # captions in another language, so fall back to whatever's actually available rather
        # than failing outright just because it isn't English.
        transcript = api.fetch(video_id)
    except NoTranscriptFound:
        try:
            transcript = next(iter(api.list(video_id))).fetch()
        except Exception as exc:
            raise YoutubeExtractionError("No transcript is available for this video.") from exc
    except TranscriptsDisabled as exc:
        raise YoutubeExtractionError("Transcripts are disabled for this video.") from exc
    except Exception as exc:  # noqa: BLE001 - surface any other API failure as a clean message
        raise YoutubeExtractionError(f"Could not fetch transcript: {exc}") from exc

    text = " ".join(snippet.text for snippet in transcript.snippets if snippet.text)
    return text, f"YouTube: {video_id}"
