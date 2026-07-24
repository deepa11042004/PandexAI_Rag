"""Website scraping via requests + BeautifulSoup."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 15
# A self-identifying UA ("RAGDocumentAssistant/1.0") gets served bot-fallback pages ("please
# upgrade your browser") by plenty of real sites (Canva included) that gate on UA sniffing - a
# real browser UA + the Accept headers a browser would normally send avoids that.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")
_MIN_TEXT_LENGTH = 200  # below this, treat as "nothing usable" rather than ingest a near-empty page


class WebsiteExtractionError(Exception):
    pass


def scrape(url: str) -> tuple[str, str]:
    """Return (page_text, page_title) for a website URL."""
    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebsiteExtractionError(f"Could not fetch URL: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines), title
