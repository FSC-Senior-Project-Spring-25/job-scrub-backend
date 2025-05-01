import os
from contextlib import asynccontextmanager

import aiohttp
import boto3
import firebase_admin
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi_injectable import register_app, cleanup_all_exit_stacks
from firebase_admin import credentials
from pinecone import Pinecone
from starlette.middleware.cors import CORSMiddleware

from context import RequestContextMiddleware
from dependencies import S3, Firestore
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.follows import router as follows_router
from routes.jobs import router as jobs_router
from routes.posts import router as posts_router
from routes.resume import router as resume_router
from routes.user_search import router as user_search_router
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.supervisor_agent import SupervisorAgent
from services.agents.user_profile_agent import UserProfileAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobsPostingService
from services.jobs_verification import JobsVerificationService
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder

load_dotenv()

# Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("job-postings")
resumes_index = pc.Index("resumes")

# S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-2"),
    config=Config(signature_version="s3v4")
)

BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register app with FastAPI Injectable
    await register_app(app)

    # Initialize Firebase Admin SDK
    cred = credentials.Certificate("./firebase.json")
    firebase_app = firebase_admin.initialize_app(cred)

    # Initialize dependencies
    session = aiohttp.ClientSession()
    s3 = S3(BUCKET_NAME, s3_client)
    firestore = Firestore(firebase_app)
    embedder = TextEmbedder()
    gemini_llm = GeminiLLM()

    job_posting_service = JobsPostingService(embedder, index, gemini_llm, session)
    job_verification_service = JobsVerificationService(session, index, embedder)

    resume_parser = ResumeParser()
    resume_matching_agent = ResumeMatchingAgent(
        resume_parser=resume_parser,
        text_embedder=embedder,
    )

    app.state.session = session
    app.state.s3_service = s3
    app.state.firestore = firestore
    app.state.pinecone = pc
    app.state.embedder = embedder
    app.state.resume_parser = resume_parser
    app.state.job_posting_service = job_posting_service
    app.state.job_verification_service = job_verification_service
    app.state.gemini_llm = gemini_llm
    app.state.resume_agent = resume_matching_agent

    yield
    # Cleanup resources
    await cleanup_all_exit_stacks()


app = FastAPI(lifespan=lifespan)

# middleware to set request context
app.add_middleware(RequestContextMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # TODO change this to deployed frontend URL in future
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["set-cookie"]
)

# Include routers
app.include_router(jobs_router, prefix="/job", tags=["jobs"])
app.include_router(resume_router, prefix="/resume", tags=["resume"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(posts_router, prefix="/posts", tags=["posts"])
app.include_router(auth_router, prefix="/auth", tags=["auth"])

app.include_router(user_search_router, prefix="/users")

app.include_router(follows_router, prefix="/users", tags=["follows"])

