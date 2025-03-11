from typing import Annotated

from fastapi import Request, Depends

from context import request_context
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobPostingService
from services.resume_parser import ResumeParser
from google.cloud import firestore
from services.text_embedder import TextEmbedder
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from fastapi import HTTPException, Request
from google.cloud import firestore as google_firestore

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

class FirestoreDB:
    def __init__(self):
        # If your credentials are set via environment variable 
        # (GOOGLE_APPLICATION_CREDENTIALS), this will just work.
        self.db = firestore.Client()

    def collection(self, name: str):
        return self.db.collection(name)

async def get_db():
    return FirestoreDB()
try:
    # If already initialized (e.g., in dev with reload), this won't run again.
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)

