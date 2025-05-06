from typing import Annotated

from aiohttp import ClientSession
from fastapi import Request, Depends, HTTPException
from firebase_admin.auth import verify_id_token
from pinecone import Pinecone

from context import request_context
from models.user import User
from services.firestore import FirestoreDB
from services.jobs_posting import JobsPostingService
from services.jobs_verification import JobsVerificationService
from services.resume_parser import ResumeParser
from services.s3 import S3Service
from services.text_embedder import TextEmbedder


async def get_request_context() -> Request:
    """Get request context from FastAPI app"""
    return request_context.get()


async def get_session(request: Request) -> ClientSession:
    """Create and provide an async HTTP client session."""
    return request.app.state.session


async def get_current_user(request: Request) -> User:
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
        decoded_token = verify_id_token(token, check_revoked=True, clock_skew_seconds=10)
        return User(
            user_id=decoded_token["uid"],
            email=decoded_token["email"],
        )
    except Exception as e:
        print(f"Invalid authentication token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {str(e)}"
        )


async def get_s3_service(request: Request) -> S3Service:
    """Get S3 service from app state"""
    return request.app.state.s3_service


async def get_firestore(request: Request):
    """ Get Firestore DB from app state """
    return request.app.state.firestore


async def get_pinecone(request: Request) -> Pinecone:
    """Get Pinecone client from app state"""
    return request.app.state.pinecone


async def get_embedder(request: Request) -> TextEmbedder:
    """Get embedder from app state"""
    return request.app.state.embedder


async def get_job_posting_service(request: Request) -> JobsPostingService:
    """Get embedder from app state"""
    return request.app.state.job_posting_service


async def get_job_verification_service(request: Request) -> JobsVerificationService:
    """Get embedder from app state"""
    return request.app.state.job_verification_service


async def get_resume_parser(request: Request) -> ResumeParser:
    """Get embedder from app state"""
    return request.app.state.resume_parser


# Type annotations for dependency injection (used in non-FastAPI routes with @injectable)
CurrentUser = Annotated[User, Depends(get_current_user)]
S3 = Annotated[S3Service, Depends(get_s3_service)]
Firestore = Annotated[FirestoreDB, Depends(get_firestore)]
PineconeClient = Annotated[Pinecone, Depends(get_pinecone)]
Embedder = Annotated[TextEmbedder, Depends(get_embedder)]
JobPoster = Annotated[JobsPostingService, Depends(get_job_posting_service)]
JobVerifier = Annotated[JobsVerificationService, Depends(get_job_verification_service)]
Parser = Annotated[ResumeParser, Depends(get_resume_parser)]
