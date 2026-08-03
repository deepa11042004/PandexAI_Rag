"""Shared chunking/cleaning logic and the file-extension dispatcher for all loaders."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from utils.helpers import clean_text

FILE_EXTENSION_TYPES = {
    "pdf": "pdf",
    "docx": "docx",
    "txt": "txt",
    "csv": "csv",
    "xlsx": "xlsx",
    "xls": "xlsx",
    "pptx": "pptx",
    "md": "markdown",
    "markdown": "markdown",
    "json": "json",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "webp": "image",
}
SUPPORTED_EXTENSIONS = tuple(FILE_EXTENSION_TYPES.keys())


class UnsupportedFileTypeError(Exception):
    pass


@dataclass
class LoadResult:
    """Outcome of loading one source (file, website, or YouTube video)."""

    name: str
    source_type: str
    chunks: list[Document] = field(default_factory=list)
    char_count: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.chunks) > 0


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def build_chunks(raw_text: str, name: str, source_type: str, extra_metadata: dict | None = None) -> LoadResult:
    """Clean and split raw extracted text into Document chunks with citation metadata."""
    text = clean_text(raw_text)
    if not text:
        return LoadResult(name=name, source_type=source_type, error="No extractable text found.")

    pieces = _splitter().split_text(text)
    metadata_base = {"source": name, "source_type": source_type, **(extra_metadata or {})}
    chunks = [
        Document(
            page_content=piece,
            metadata={**metadata_base, "chunk_index": idx, "total_chunks": len(pieces)},
        )
        for idx, piece in enumerate(pieces)
    ]
    return LoadResult(name=name, source_type=source_type, chunks=chunks, char_count=len(text))


def load_file(filename: str, raw_bytes: bytes) -> LoadResult:
    """Extract, clean, and chunk an uploaded file based on its extension."""
    # Imported lazily to avoid circular imports and to keep optional OCR deps out of the hot path
    # for users who never upload images.
    from loaders import image_loader, office_loaders, text_loaders

    extension = get_extension(filename)
    source_type = FILE_EXTENSION_TYPES.get(extension)
    if source_type is None:
        raise UnsupportedFileTypeError(
            f"'.{extension}' is not supported. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    extractors = {
        "pdf": office_loaders.parse_pdf,
        "docx": office_loaders.parse_docx,
        "pptx": office_loaders.parse_pptx,
        "csv": office_loaders.parse_csv,
        "xlsx": office_loaders.parse_xlsx,
        "txt": text_loaders.parse_txt,
        "markdown": text_loaders.parse_markdown,
        "json": text_loaders.parse_json,
        "image": image_loader.parse_image,
    }
    raw_text = extractors[source_type](raw_bytes)
    return build_chunks(raw_text, name=filename, source_type=source_type)


def load_website(url: str) -> LoadResult:
    from loaders.website_loader import scrape

    raw_text, title = scrape(url)
    return build_chunks(raw_text, name=title, source_type="website", extra_metadata={"url": url})


def load_youtube(url: str) -> LoadResult:
    from loaders.youtube_loader import fetch_transcript

    raw_text, title = fetch_transcript(url)
    return build_chunks(raw_text, name=title, source_type="youtube", extra_metadata={"url": url})
