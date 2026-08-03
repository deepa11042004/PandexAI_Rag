"""LangGraph orchestration for the RAG pipeline's *reasoning* steps (retrieval through the final
prompt). Token-by-token generation happens after the graph, in `rag_pipeline.stream_answer`, so
the client can see tokens live while retrieval/citations are already resolved.

Pipeline: rewrite_query -> retrieve -> rerank -> decide (has_context?) -> build_prompt
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import retriever
from config import get_settings
from llm_provider import build_llm, fallback_order
from vector_store import VectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the \
information contained in the CONTEXT below, which was retrieved from documents the user uploaded.

Rules:
- Do not use any outside knowledge, even if you are confident it is correct.
- If the context does not contain enough information to answer, respond with EXACTLY this \
sentence and nothing else: "{not_found}"
- When you do answer, be concise and accurate, and stay strictly grounded in the context.
- Do not mention "the context" or "the documents" explicitly; answer naturally as if you already \
knew the information.

CONTEXT:
{context}"""

REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up \
into a standalone question that contains all the context needed to search a document database. \
If the follow-up is already standalone, return it unchanged. Return ONLY the rewritten question.

History:
{history}

Follow-up question: {question}
Standalone question:"""


class GraphState(TypedDict):
    question: str
    chat_history: list[tuple[str, str]]
    standalone_question: str
    vector_store: Any  # VectorStoreManager - kept as Any to avoid TypedDict/pydantic coupling
    provider_hint: str | None
    rerank_top_k: int | None
    source_ids: list[str]
    retrieved_pairs: list
    chunks: list  # list[retriever.RetrievedChunk]
    context: str
    citations: list[str]
    has_context: bool
    messages: list[BaseMessage]


def _rewrite_query(state: GraphState) -> dict:
    """Turn a follow-up question into a standalone one, using recent chat history."""
    if not state["chat_history"]:
        return {"standalone_question": state["question"]}

    candidates = fallback_order(state.get("provider_hint"))
    if not candidates:
        return {"standalone_question": state["question"]}

    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in state["chat_history"][-4:])
    prompt = REWRITE_PROMPT.format(history=history_text, question=state["question"])

    for config in candidates:
        try:
            llm = build_llm(config, temperature=0.0, streaming=False)
            result = llm.invoke([HumanMessage(content=prompt)])
            rewritten = (result.content or "").strip()
            return {"standalone_question": rewritten or state["question"]}
        except Exception as exc:  # noqa: BLE001 - rewriting is best-effort, never block the pipeline
            logger.warning("Query rewrite failed via '%s': %s", config.name, exc)
            continue

    return {"standalone_question": state["question"]}


def _retrieve(state: GraphState) -> dict:
    manager: VectorStoreManager = state["vector_store"]
    pairs = retriever.vector_search_per_source(manager, state["standalone_question"], state["source_ids"])
    return {"retrieved_pairs": pairs}


def _rerank(state: GraphState) -> dict:
    if not state["retrieved_pairs"]:
        return {"chunks": [], "has_context": False, "context": "", "citations": []}

    chunks = retriever.hybrid_rerank(state["retrieved_pairs"], state["standalone_question"], top_k=state.get("rerank_top_k"))
    has_context = len(chunks) > 0
    context = retriever.format_context(chunks) if has_context else ""
    citations = retriever.unique_sources(chunks) if has_context else []
    return {"chunks": chunks, "has_context": has_context, "context": context, "citations": citations}


def _build_prompt(state: GraphState) -> dict:
    if not state["has_context"]:
        return {"messages": []}

    settings = get_settings()
    system = SystemMessage(content=SYSTEM_PROMPT.format(not_found=NOT_FOUND_MESSAGE, context=state["context"]))
    messages: list[BaseMessage] = [system]
    for question, answer in state["chat_history"][-settings.max_history_turns :]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    messages.append(HumanMessage(content=state["question"]))
    return {"messages": messages}


def _route_after_rerank(state: GraphState) -> str:
    return "build_prompt" if state["has_context"] else END


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("build_prompt", _build_prompt)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", _route_after_rerank, {"build_prompt": "build_prompt", END: END})
    graph.add_edge("build_prompt", END)
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_retrieval_graph(
    question: str,
    chat_history: list[tuple[str, str]],
    vector_store: VectorStoreManager,
    provider_hint: str | None,
    rerank_top_k: int | None = None,
    source_ids: list[str] | None = None,
) -> GraphState:
    """Run the graph end-to-end and return the final state (context/citations/messages)."""
    graph = get_compiled_graph()
    initial: GraphState = {
        "question": question,
        "chat_history": chat_history,
        "standalone_question": "",
        "vector_store": vector_store,
        "provider_hint": provider_hint,
        "rerank_top_k": rerank_top_k,
        "source_ids": source_ids or [],
        "retrieved_pairs": [],
        "chunks": [],
        "context": "",
        "citations": [],
        "has_context": False,
        "messages": [],
    }
    return await graph.ainvoke(initial)
