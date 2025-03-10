from typing import Annotated

from aiohttp import ClientSession
from fastapi import Request, Depends
from fastapi import Request, Depends, HTTPException
from firebase_admin.auth import verify_id_token

from context import request_context
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobPostingService
from services.resume_parser import ResumeParser
from services.s3 import S3Service

from services.text_embedder import TextEmbedder


async def get_request_context() -> Request:
    """Get request context from FastAPI app"""
    return request_context.get()


async def get_session(request: Request) -> ClientSession:
    """Create and provide an async HTTP client session."""
    return request.app.state.session


async def get_current_user(request: Request) -> dict[str, str]:
    """
    Verify Firebase ID token from Authorization header and return user info
    """
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )

    token = authorization.split("Bearer ")[1]
    try:
        # Verify the Firebase ID token
        decoded_token = verify_id_token(token)
        return {
            "user_id": decoded_token["uid"],
            "email": decoded_token.get("email", "")
        }
    except Exception as e:
        print(f"Invalid authentication token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {str(e)}"
        )


async def get_s3_service(request: Request) -> S3Service:
    """Get S3 service from app state"""
    return request.app.state.s3_service


async def get_embedder(request: Request) -> TextEmbedder:
    """Get embedder from app state"""
    return request.app.state.embedder


async def get_job_service(request: Request) -> JobPostingService:
    """Get embedder from app state"""
    return request.app.state.job_service


async def get_gemini_llm(request: Request) -> GeminiLLM:
    """Get embedder from app state"""
    return request.app.state.gemini_llm


async def get_resume_parser(request: Request) -> ResumeParser:
    """Get embedder from app state"""
    return request.app.state.resume_parser


async def get_resume_agent(request: Request) -> ResumeMatchingAgent:
    """Get embedder from app state"""
    return request.app.state.resume_agent


# Type annotations for dependency injection (used in non-FastAPI routes with @injectable)
S3 = Annotated[S3Service, Depends(get_s3_service)]
Embedder = Annotated[TextEmbedder, Depends(get_embedder)]
JobService = Annotated[TextEmbedder, Depends(get_job_service)]
GeminiLLM = Annotated[GeminiLLM, Depends(get_gemini_llm)]
ResumeParser = Annotated[ResumeParser, Depends(get_resume_parser)]
MatchingAgent = Annotated[ResumeMatchingAgent, Depends(get_resume_agent)]
