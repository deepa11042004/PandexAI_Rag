"""Pydantic request/response schemas shared by the FastAPI endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    id: str
    name: str
    type: str  # pdf, docx, txt, csv, xlsx, pptx, image, youtube, website, markdown, json
    chunks: int
    size_bytes: int = 0
    added_at: str


class SourceListResponse(BaseModel):
    sources: list[SourceInfo]
    count: int


class AddUrlRequest(BaseModel):
    url: str = Field(..., description="Website URL to scrape and ingest")


class AddYoutubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL to transcribe and ingest")


class IngestResponse(BaseModel):
    success: bool
    source: SourceInfo | None = None
    message: str = ""
    duplicate: bool = False


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    citations: list[str] = Field(default_factory=list)
    timestamp: str = ""


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    provider: str = ""
    model: str = ""


class SessionSettings(BaseModel):
    llm_provider: str | None = None
    chat_model: str | None = None
    retrieval_top_k: int | None = None
    rerank_top_k: int | None = None
    mcp_math_enabled: bool | None = None


class TranscriptionResponse(BaseModel):
    text: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    pinned: bool = False
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    message_count: int = 0
    duration_seconds: float | None = None


class ConversationDetail(ConversationSummary):
    messages: list[ChatMessage] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    total: int
    page: int
    page_size: int
    has_more: bool
    available_models: list[str] = Field(default_factory=list)


class ConversationUpdateRequest(BaseModel):
    pinned: bool | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
