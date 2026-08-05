"""Chat REST + SSE-streaming endpoints, backed entirely by MongoDB (via services.chat_service).

    POST   /chat              - send a message; creates the chat automatically if needed (SSE)
    GET    /chats/{user_id}    - list a user's chats
    GET    /chat/{chat_id}     - full conversation (all messages)
    PATCH  /chat/{chat_id}     - rename a chat's title
    DELETE /chat/{chat_id}     - delete a chat and all of its messages
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from database.models import ChatDetail, ChatListResponse, ChatMessageRequest, ChatSummary, RenameChatRequest
from database.mongodb import MongoConnectionError, get_database
from database.repository import ChatRepository, MessageRepository, RepositoryError, UserRepository
from services.chat_service import ChatNotFoundError, ChatService, EmptyMessageError, InvalidChatIdError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


def get_chat_service() -> ChatService:
    """FastAPI dependency: builds a request-scoped ChatService wired to the Mongo repositories."""
    try:
        db = get_database()
    except MongoConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatService(UserRepository(db), ChatRepository(db), MessageRepository(db))


def _handle_repository_error(exc: RepositoryError) -> None:
    logger.exception("Repository error")
    raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat")
async def send_chat_message(
    payload: ChatMessageRequest, service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    """Server-Sent Events stream: a `meta` event with the resolved chat_id/title, then one
    `data: {...}` line per token, then a final `done` event."""

    async def event_stream():
        try:
            async for event in service.send_message(payload.user_id, payload.chat_id, payload.question):
                yield f"data: {json.dumps(event)}\n\n"
        except EmptyMessageError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except InvalidChatIdError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except RepositoryError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Database error: {exc}'})}\n\n"
        except Exception as exc:  # noqa: BLE001 - never let an unhandled error hang the stream
            logger.exception("Unhandled error streaming chat for user %s", payload.user_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/chats/{user_id}", response_model=ChatListResponse)
async def list_chats(user_id: str, service: ChatService = Depends(get_chat_service)) -> ChatListResponse:
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    try:
        chats = await service.list_chats(user_id)
    except RepositoryError as exc:
        _handle_repository_error(exc)
    return ChatListResponse(chats=chats)


@router.get("/chat/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: str, service: ChatService = Depends(get_chat_service)) -> ChatDetail:
    try:
        return await service.get_chat_detail(chat_id)
    except InvalidChatIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RepositoryError as exc:
        _handle_repository_error(exc)


@router.patch("/chat/{chat_id}", response_model=ChatSummary)
async def rename_chat(
    chat_id: str, payload: RenameChatRequest, service: ChatService = Depends(get_chat_service)
) -> ChatSummary:
    try:
        return await service.rename_chat(chat_id, payload.title)
    except InvalidChatIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyMessageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RepositoryError as exc:
        _handle_repository_error(exc)


@router.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str, service: ChatService = Depends(get_chat_service)) -> dict:
    try:
        await service.delete_chat(chat_id)
    except InvalidChatIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RepositoryError as exc:
        _handle_repository_error(exc)
    return {"success": True}


__all__ = ["router"]
