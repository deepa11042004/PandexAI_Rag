"""MongoDB-backed persistence layer: connection, Pydantic models, and the repository pattern
for the `users`, `chats`, and `messages` collections (see mongodb.py / models.py / repository.py).

This is the ONLY persistent store for chat/message history - no in-memory cache, session state,
or LangGraph checkpointer is used for that data.
"""
