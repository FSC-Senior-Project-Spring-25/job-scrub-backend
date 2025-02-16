from typing import Annotated

from fastapi import Request, Depends

from context import request_context
from services.text_embedder import TextEmbedder


async def get_request_context() -> Request:
    """Get request context from FastAPI app"""
    return request_context.get()


async def get_embedder(request: Request) -> TextEmbedder:
    """Get embedder from app state"""
    return request.app.state.embedder


async def get_job_service(request: Request) -> TextEmbedder:
    """Get embedder from app state"""
    return request.app.state.job_service

# Type annotations for dependency injection (used in non-FastAPI routes with @injectable)
Embedder = Annotated[TextEmbedder, Depends(get_embedder)]
JobService = Annotated[TextEmbedder, Depends(get_job_service)]
