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
    app.state.job_service = JobPostingService(app.state.embedder)

    yield
    # Cleanup resources
    del app.state.embedder
    del app.state.job_service
    del app.state.index
    await cleanup_all_exit_stacks()

app = FastAPI(lifespan=lifespan)

# middleware to set request context
app.add_middleware(RequestContextMiddleware)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
