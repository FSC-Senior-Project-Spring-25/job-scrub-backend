import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi_injectable import register_app, cleanup_all_exit_stacks
from pinecone import Pinecone
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from context import RequestContextMiddleware
from dependencies import get_job_service, get_resume_agent, get_current_user
from models.job_report import JobReport
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobPostingService
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

    # Initialize dependencies
    embedder = TextEmbedder()
    job_posting_service = JobPostingService(embedder, index)

    resume_parser = ResumeParser()
    gemini_llm = GeminiLLM()
    resume_matching_agent = ResumeMatchingAgent(
        resume_parser=resume_parser,
        text_embedder=embedder,
        llm=gemini_llm,
    )

    app.state.embedder = embedder
    app.state.job_service = job_posting_service
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
async def create_job_report(report: JobReport, job_service: JobPostingService = Depends(get_job_service)):
    try:
        id = await job_service.post_job(report)
        return {"message": "Job report created successfully with ID: " + id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume/match")
async def calculate_resume_similarity(
        resume_file: Annotated[UploadFile, File(alias="resumeFile", validation_alias="resumeFile")],
        job_description: str = Form(
            ...,
            alias="jobDescription",
            validation_alias="jobDescription",
            min_length=1,
            max_length=5000
        ),
        matching_agent: ResumeMatchingAgent = Depends(get_resume_agent)
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
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_id = current_user["user_id"]

    # Create a unique filename that includes the user ID to enforce ownership
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_filename = f"resumes/{user_id}/{timestamp}-{uuid.uuid4()}.pdf"

    try:
        # Read file content
        file_content = await file.read()

        if len(file_content) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")

        # Upload file to S3 with metadata to track ownership
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=unique_filename,
            Body=file_content,
            ContentType='application/pdf',
            Metadata={
                'user_id': user_id
            }
        )

        # Return a reference ID or the S3 key itself
        # Do NOT return a public URL since the bucket blocks public access
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_id": unique_filename,  # This is the internal reference to use later
        })

    except ClientError as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to upload file to S3")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/view-resume")
async def view_resume(
        key: str,
        current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    if not key.startswith(f"resumes/{user_id}/"):
        raise HTTPException(status_code=403, detail="Not authorized to access this file")

    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': key
            },
            ExpiresIn=3600
        )

        return JSONResponse(content={"url": url})

    except ClientError as e:
        print(f"S3 error details: {str(e)}")
        raise HTTPException(status_code=404, detail="File not found or access denied")
