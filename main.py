import os
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi_injectable import register_app, cleanup_all_exit_stacks
from pinecone import Pinecone

from context import RequestContextMiddleware
from dependencies import get_job_service, get_resume_agent
from models.job_report import JobReport
from services.agents.resume_matcher import ResumeMatchingAgent
from services.gemini import GeminiLLM
from services.jobs_posting import JobPostingService
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder
from services.posts import router as posts_router #Importing the posts route

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("job-postings")


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
app.include_router(posts_router, prefix="/api", tags=["posts"])

# middleware to set request context
app.add_middleware(RequestContextMiddleware)


@app.get("/")
async def root():
    return {"message": "Hello World"}


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
