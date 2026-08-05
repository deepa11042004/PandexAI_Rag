"""FastAPI routers. Route handlers stay thin: parse/validate input, call the service layer,
translate domain errors into HTTP status codes. No MongoDB logic lives here - see
`backend/database/` and `backend/services/chat_service.py`.
"""
