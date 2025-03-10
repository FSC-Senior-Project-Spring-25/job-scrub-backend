import os
from contextlib import asynccontextmanager
from typing import Annotated

import aiohttp
import boto3
import firebase_admin
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi_injectable import register_app, cleanup_all_exit_stacks
from firebase_admin import credentials
from pinecone import Pinecone
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from context import RequestContextMiddleware
from dependencies import get_current_user, S3, MatchingAgent, JobPostingService, JobVerificationService
from models.job_report import JobReport
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobsPostingService
from services.jobs_verification import JobsVerificationService
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder

load_dotenv()

# Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("job-postings")

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
    firebase_admin.initialize_app(cred)

    # Initialize dependencies
    session = aiohttp.ClientSession()
    s3 = S3(BUCKET_NAME, s3_client)
    S3(BUCKET_NAME, s3_client)
    embedder = TextEmbedder()
    job_posting_service = JobsPostingService(embedder, index, session)
    job_verification_service = JobsVerificationService(session, index, embedder)

    resume_parser = ResumeParser()
    gemini_llm = GeminiLLM()
    resume_matching_agent = ResumeMatchingAgent(
        resume_parser=resume_parser,
        text_embedder=embedder,
        llm=gemini_llm,
    )

    app.state.session = session
    app.state.s3_service = S3(BUCKET_NAME, s3_client)
    app.state.s3_service = s3
    app.state.embedder = embedder
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
)


@app.post("/job/report")
async def create_job_report(report: JobReport, job_service: JobPostingService):
    id = await job_service.post_job(report)
    return {"message": "Job report created successfully with ID: " + id}


@app.patch("/job/verify/{job_id}")
async def verify_job(job_id: str, verified: bool, report: JobReport, job_service: JobVerificationService):
    await job_service.verify_job(job_id, verified, report)
    return {"message": "Job verified successfully"}


@app.delete("/job/delete/{job_id}")
async def delete_job(job_id: str, job_service: JobVerificationService):
    await job_service.delete_job(job_id)
    return {"message": "Job deleted successfully"}


@app.get("/job/unverified")
async def get_unverified_jobs(job_service: JobVerificationService):
    jobs = await job_service.get_unverified_jobs()
    return jobs


@app.post("/resume/match")
async def calculate_resume_similarity(
        matching_agent: MatchingAgent,
        resume_file: Annotated[UploadFile, File(alias="resumeFile", validation_alias="resumeFile")],
        job_description: str = Form(
            ...,
            alias="jobDescription",
            validation_alias="jobDescription",
            min_length=1,
            max_length=5000
        ),
):
    try:
        # Validate file type
        if not resume_file.content_type == "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        # Read the PDF bytes
        file_bytes = await resume_file.read()

        # Process using the matching agent
        result = await matching_agent.analyze_resume(file_bytes, job_description)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-resume")
async def upload_resume(
        s3_service: S3,
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user),
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_id = current_user["user_id"]  # Firebase UID

    unique_filename = await s3_service.upload_file(file, user_id)

    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "file_id": unique_filename,  # This is the internal reference to use later
    })


@app.get("/view-resume")
async def view_resume(
        s3_service: S3,
        key: str,
        current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]  # Firebase UID

    # Security check: Ensure the user can only access their own files
    if not key.startswith(f"resumes/{user_id}/"):
        raise HTTPException(status_code=403, detail="Not authorized to access this file")

    url = s3_service.get_presigned_url(key)
    return JSONResponse(content={"url": url})
