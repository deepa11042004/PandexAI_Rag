"""Real MCP client that routes math questions to an external MCP math server
(e.g. Calculator_MCP_Server) over stdio, with graceful fallback for stats
queries the external server doesn't expose as tools."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import statistics
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_MATH_KEYWORDS = {
    "add", "plus", "sum", "subtract", "minus", "difference", "times", "multiply", "product",
    "divide", "quotient", "mod", "modulus", "power", "exponent", "square root", "sqrt",
    "cube root", "cuberoot", "log", "logarithm", "sin", "cos", "tan", "average", "mean",
    "median", "mode", "percentage", "percent", "factorial", "matrix", "equation", "solve",
    "algebra", "statistics", "standard deviation", "variance", "unit", "convert", "conversion",
    "financial", "scientific", "calculate", "compute", "total", "ratio",
}

_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

# Stat operations the external MCP math server has no tool for - computed locally
# from numbers found in the query/retrieved context instead of round-tripping.
_STATS_KEYWORDS = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "mode": "mode",
    "standard deviation": "stdev",
    "variance": "variance",
}

_WORD_OPERATORS = (
    (re.compile(r"to the power of|raised to", re.IGNORECASE), "**"),
    (re.compile(r"multiplied by", re.IGNORECASE), "*"),
    (re.compile(r"divided by", re.IGNORECASE), "/"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bmodulus\b|\bmod\b", re.IGNORECASE), "%"),
)

_EXPRESSION_TOKEN = re.compile(
    r"sqrt|log|sin|cos|tan|asin|acos|atan|pi|[0-9]*\.[0-9]+|[0-9]+|\*\*|[+\-*/%()]",
    re.IGNORECASE,
)


def classify_query(question: str) -> str:
    """Classify the incoming question as rag, math, or mixed."""
    text = (question or "").strip().lower()
    if not text:
        return "rag"

    if any(token in text for token in ("+", "-", "*", "/", "^", "=", "sqrt", "log", "sin", "cos", "tan", "×", "÷")):
        return "math"

    if any(keyword in text for keyword in _MATH_KEYWORDS):
        if any(word in text for word in ("document", "pdf", "docx", "txt", "page", "uploaded", "source", "file")):
            return "mixed"
        return "math"

    return "rag"


def extract_numeric_values(context: str) -> list[float]:
    """Extract numeric values from retrieved document content."""
    if not context:
        return []
    return [float(value) for value in _NUMERIC_PATTERN.findall(context)]


def _match_stats_op(text: str) -> str | None:
    for phrase, op in _STATS_KEYWORDS.items():
        if phrase in text:
            return op
    return None


def _compute_stats(op: str, numbers: list[float]) -> str:
    if not numbers:
        return "No numeric values were found to compute that."
    try:
        if op == "mean":
            return f"The mean is {statistics.mean(numbers):.4f}."
        if op == "median":
            return f"The median is {statistics.median(numbers):.4f}."
        if op == "mode":
            return f"The mode is {statistics.mode(numbers):.4f}."
        if op == "stdev":
            if len(numbers) < 2:
                return "Need at least two values to compute standard deviation."
            return f"The standard deviation is {statistics.pstdev(numbers):.4f}."
        if op == "variance":
            if len(numbers) < 2:
                return "Need at least two values to compute variance."
            return f"The variance is {statistics.pvariance(numbers):.4f}."
    except statistics.StatisticsError as exc:
        return str(exc)
    return "Unsupported statistical operation."


def _build_expression(text: str) -> str:
    """Turn a natural-language math question into an expression string that
    Calculator_MCP_Server's `evaluate_expression` tool can parse."""
    normalized = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    for pattern, replacement in _WORD_OPERATORS:
        normalized = pattern.sub(replacement, normalized)
    tokens = _EXPRESSION_TOKEN.findall(normalized)
    return "".join(tokens)


def _parse_tool_result(result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        text = result.content[0].text if result.content else "Unknown MCP tool error."
        return {"success": False, "error": text}

    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured

    if result.content:
        text = getattr(result.content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"success": True, "result": text}

    return {"success": False, "error": "Empty response from the MCP math server."}


class MCPMathClient:
    """Keeps a persistent MCP session open to an external stdio math server
    (Calculator_MCP_Server by default) and calls its tools on demand."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: str | None = None

    def _server_params(self) -> StdioServerParameters:
        command = self.settings.mcp_math_server_command.strip()
        if not command:
            raise RuntimeError(
                "MCP_MATH_SERVER_COMMAND is not configured. Point it at your Calculator_MCP_Server "
                "(see .env.example)."
            )
        raw_args = self.settings.mcp_math_server_args.strip()
        # posix=False: on Windows these are filesystem paths, and posix-mode shlex
        # treats backslashes as escape characters and mangles them.
        args = shlex.split(raw_args, posix=False) if raw_args else []
        return StdioServerParameters(command=command, args=args)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._session is not None:
                return
            self._ready.clear()
            self._start_error = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

        if not self._ready.wait(timeout=self.settings.mcp_math_timeout_seconds):
            raise RuntimeError("Timed out starting the MCP math server.")
        if self._start_error:
            raise RuntimeError(self._start_error)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._session_main())
        except Exception as exc:  # noqa: BLE001
            self._start_error = str(exc)
            self._ready.set()
        finally:
            loop.close()

    async def _session_main(self) -> None:
        params = self._server_params()
        self._stop_event = asyncio.Event()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop_event.wait()
        finally:
            self._session = None

    def _restart(self) -> None:
        with self._lock:
            if self._loop is not None and self._stop_event is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._stop_event.set)
            self._thread = None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_started()
        assert self._loop is not None and self._session is not None
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        try:
            result = future.result(timeout=self.settings.mcp_math_timeout_seconds)
        except Exception:
            self._restart()
            raise
        return _parse_tool_result(result)

    def invoke(self, query: str, context: str | None = None) -> str:
        """Answer a math or mixed query via the MCP math server, falling back to
        local statistics for aggregate queries the server has no tool for."""
        text = (query or "").strip()
        low = text.lower()
        try:
            stats_op = _match_stats_op(low)
            if stats_op:
                numbers = extract_numeric_values(context or "") or extract_numeric_values(text)
                return _compute_stats(stats_op, numbers)

            expression = _build_expression(text)
            if not expression:
                return "I couldn't find a valid mathematical expression in that question."

            payload = self.call_tool("evaluate_expression", {"expression": expression})
            if not payload.get("success", False):
                return payload.get("error") or "The math server could not compute that request."
            return f"{expression} = {payload.get('result')}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP math server unavailable: %s", exc)
            return (
                "The math server is currently unavailable. Please try a simpler calculation or ask "
                "me to answer from the uploaded documents instead."
            )


_math_client = MCPMathClient()


async def evaluate_math_query(query: str, context: str | None = None) -> str | None:
    """Evaluate a math or mixed query through the MCP server."""
    try:
        return await asyncio.to_thread(_math_client.invoke, query, context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Math evaluation failed: %s", exc)
        return None
