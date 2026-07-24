"""Chat LLM provider abstraction with automatic Groq -> OpenRouter -> OpenAI -> Claude -> Gemini
fallback.

Groq, OpenRouter, and OpenAI all expose an OpenAI-compatible Chat Completions API, so a single
`ChatOpenAI` client (pointed at a different `base_url`) works for all three - no need for
provider-specific SDKs. Claude (Anthropic) and Gemini (Google) are NOT OpenAI-compatible and use
their own LangChain clients (`ChatAnthropic`/`ChatGoogleGenerativeAI`) - see `build_llm`'s `kind`
dispatch below. All of them implement the same LangChain `BaseChatModel` streaming interface, so
`astream_with_fallback` doesn't need to know or care which kind of client it's holding.

Embeddings are separate: none of the chat-only providers (Groq/OpenRouter/Anthropic/Gemini) offer
an embeddings endpoint compatible with what this app expects, so `get_embeddings_client()` defaults
to a local, keyless HuggingFace sentence-transformers model (EMBEDDING_PROVIDER=local) and only
needs OPENAI_API_KEY if you explicitly opt into EMBEDDING_PROVIDER=openai.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AuthenticationError, OpenAI, OpenAIError

from backend.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

ProviderKind = Literal["openai_compatible", "anthropic", "google"]


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: str | None = None  # None => provider's default endpoint (OpenAI-compatible kinds only)
    kind: ProviderKind = "openai_compatible"


def _provider_configs() -> list[ProviderConfig]:
    settings = get_settings()
    return [
        ProviderConfig("groq", settings.groq_api_key, settings.groq_chat_model, settings.groq_base_url),
        ProviderConfig(
            "openrouter", settings.openrouter_api_key, settings.openrouter_chat_model, settings.openrouter_base_url
        ),
        ProviderConfig("openai", settings.openai_api_key, settings.openai_chat_model),
        ProviderConfig("anthropic", settings.anthropic_api_key, settings.anthropic_chat_model, kind="anthropic"),
        ProviderConfig("google", settings.google_api_key, settings.google_chat_model, kind="google"),
    ]


def fallback_order(preferred: str | None) -> list[ProviderConfig]:
    """Preferred provider first (if it has a key), then the remaining configured providers."""
    configs = _provider_configs()
    configured = [c for c in configs if c.api_key]
    if not preferred:
        return configured
    preferred_configs = [c for c in configured if c.name == preferred]
    rest = [c for c in configured if c.name != preferred]
    return preferred_configs + rest


def has_any_provider_configured() -> bool:
    return any(c.api_key for c in _provider_configs())


_local_embeddings_cache: Embeddings | None = None


def _get_local_embeddings() -> Embeddings:
    """A small, free, keyless sentence-transformers model running on the local CPU.

    Cached at module level since loading the model (and its ~80MB download on first use) is
    expensive - we only want to pay that cost once per process, not once per session.
    """
    global _local_embeddings_cache
    if _local_embeddings_cache is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        settings = get_settings()
        logger.info("Loading local embedding model '%s' (first run downloads it)...", settings.local_embedding_model)
        _local_embeddings_cache = HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _local_embeddings_cache


def get_embeddings_client() -> Embeddings:
    settings = get_settings()
    if settings.embedding_provider == "local":
        return _get_local_embeddings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. Either set that key, or "
            "switch EMBEDDING_PROVIDER=local (default) to use the free local model instead."
        )
    return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Lightweight validation (models.list) for the OpenAI key used for embeddings."""
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    if not api_key.startswith("sk-"):
        return False, "API key format looks invalid (should start with 'sk-')."
    try:
        OpenAI(api_key=api_key).models.list()
        return True, "Connected to OpenAI."
    except AuthenticationError:
        return False, "Authentication failed - check your API key."
    except OpenAIError as exc:
        return False, f"Could not reach OpenAI: {exc}"


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech-to-text for the chat input's voice recorder.

    Tries Groq's Whisper endpoint first (free tier, same GROQ_API_KEY as chat), then falls back to
    OpenAI's Whisper if Groq has no key or the request fails. OpenRouter is skipped - it's a
    chat-completions proxy only and has no audio-transcription endpoint.
    """
    settings = get_settings()
    candidates = [
        ("groq", settings.groq_api_key, settings.groq_base_url, settings.groq_whisper_model),
        ("openai", settings.openai_api_key, None, settings.openai_whisper_model),
    ]
    configured = [c for c in candidates if c[1]]
    if not configured:
        raise RuntimeError(
            "No speech-to-text provider is configured. Set GROQ_API_KEY (free, "
            "https://console.groq.com/keys) or OPENAI_API_KEY."
        )

    kwargs: dict = {}
    if settings.whisper_language:
        kwargs["language"] = settings.whisper_language

    last_error: Exception | None = None
    for name, api_key, base_url, model in configured:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            transcript = client.audio.transcriptions.create(model=model, file=(filename, audio_bytes), **kwargs)
            return transcript.text
        except Exception as exc:  # noqa: BLE001 - try the next provider on any failure
            logger.warning("Speech-to-text provider '%s' failed (%s); trying next fallback.", name, exc)
            last_error = exc
            continue

    raise last_error or RuntimeError("All configured speech-to-text providers failed.")


def build_llm(config: ProviderConfig, temperature: float, streaming: bool) -> BaseChatModel:
    if config.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, api_key=config.api_key, temperature=temperature, streaming=streaming)

    if config.kind == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.model, google_api_key=config.api_key, temperature=temperature)

    kwargs: dict = {"model": config.model, "api_key": config.api_key, "temperature": temperature, "streaming": streaming}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


async def astream_with_fallback(
    messages: list[BaseMessage],
    preferred_provider: str | None = None,
    temperature: float = 0.1,
    model_override: str | None = None,
) -> AsyncIterator[tuple[str, str, str]]:
    """Stream a chat completion, trying providers in fallback order until one works.

    `model_override` (from the session settings page) replaces the model name only for the
    preferred provider's own attempt - it's meaningless for the *other* providers since a model
    id from one API generally doesn't exist on another, so fallback candidates keep their default.

    Yields (token, provider_name, model_name) tuples. The provider/model in each tuple lets the
    caller report which one actually answered. Raises the last error if every provider fails.

    Note: fallback only helps if a provider fails before yielding any tokens (the common case -
    auth/quota errors surface on the first chunk). A failure mid-stream after partial output has
    already reached the client will restart from the next provider rather than resuming.
    """
    candidates = fallback_order(preferred_provider)
    if not candidates:
        raise RuntimeError(
            "No LLM provider is configured. Set GROQ_API_KEY (free, https://console.groq.com/keys), "
            "OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
        )
    if model_override and candidates[0].name == preferred_provider:
        candidates[0] = replace(candidates[0], model=model_override)

    last_error: Exception | None = None
    for config in candidates:
        llm = build_llm(config, temperature=temperature, streaming=True)
        try:
            async for chunk in llm.astream(messages):
                yield (chunk.content or ""), config.name, config.model
            return
        except Exception as exc:  # noqa: BLE001 - try the next provider on any failure
            logger.warning("Provider '%s' failed (%s); trying next fallback.", config.name, exc)
            last_error = exc
            continue

    raise last_error or RuntimeError("All configured LLM providers failed.")
