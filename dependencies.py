from typing import Annotated

from fastapi import Request, Depends

from context import request_context
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobPostingService
from services.resume_parser import ResumeParser

from services.text_embedder import TextEmbedder


async def get_request_context() -> Request:
    """Get request context from FastAPI app"""
    return request_context.get()


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
Embedder = Annotated[TextEmbedder, Depends(get_embedder)]
JobService = Annotated[TextEmbedder, Depends(get_job_service)]
GeminiLLM = Annotated[GeminiLLM, Depends(get_gemini_llm)]
ResumeParser = Annotated[ResumeParser, Depends(get_resume_parser)]
ResumeAgent = Annotated[ResumeMatchingAgent, Depends(get_resume_agent)]
