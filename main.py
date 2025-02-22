import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi_injectable import register_app, cleanup_all_exit_stacks
from pinecone import Pinecone

from context import RequestContextMiddleware
from dependencies import get_job_service
from models.job_report import JobReport
from services.jobs_posting import JobPostingService
from services.text_embedder import TextEmbedder

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("job-postings")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register app with FastAPI Injectable
    await register_app(app)

    # Initialize embedder and job service
    app.state.embedder = TextEmbedder()
    app.state.job_service = JobPostingService(app.state.embedder, index)

    yield
    # Cleanup resources
    del app.state.embedder
    del app.state.job_service
    await cleanup_all_exit_stacks()


app = FastAPI(lifespan=lifespan)

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
