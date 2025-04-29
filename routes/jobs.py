from fastapi import APIRouter
from fastapi.params import Query

from dependencies import JobPoster, JobVerifier
from models.job_report import JobReport

router = APIRouter()


@router.post("/report")
async def create_job_report(report: JobReport, job_service: JobPoster):
    """
    Create a job report and post it to Pinecone

    Args:
        report: JobReport object containing job details
        job_service: JobPoster service
    """
    id = await job_service.post_job(report)
    return {"message": "Job report created successfully with ID: " + id}


@router.patch("/verify/{job_id}")
async def verify_job(job_id: str, verified: bool, report: JobReport, job_service: JobVerifier):
    """
    Confirm a job posting and update its metadata in Pinecone

    Args:
        job_id: The ID of the job to verify
        verified: Whether the job is verified or not
        report: JobReport object containing updated job details
        job_service: JobVerifier service
    """
    await job_service.verify_job(job_id, verified, report)
    return {"message": "Job verified successfully"}


@router.delete("/delete/{job_id}")
async def delete_job(job_id: str, job_service: JobVerifier):
    """
    Delete a job posting from Pinecone

    Args:
        job_id: The ID of the job to delete
        job_service: JobVerifier service
    """
    await job_service.delete_job(job_id)
    return {"message": "Job deleted successfully"}


@router.post("/fetch")
async def fetch_jobs(ids: list[str], job_service: JobVerifier):
    """
    Fetch multiple jobs by their IDs

    Args:
        ids: List of job IDs to fetch
        job_service: JobVerifier service

    Returns:
        Dictionary of job IDs mapped to their job data
    """
    jobs = await job_service.get_jobs(ids)
    if jobs:
        return jobs
    else:
        return {"message": "Jobs not found"}


@router.get("/unverified")
async def get_unverified_jobs(
        job_service: JobVerifier,
        limit: int = Query(1000, gt=0, le=1000)
):
    """
    Get unverified jobs from Pinecone

    Args:
        job_service: JobVerifier service
        limit: The maximum number of unverified jobs to return
    """
    jobs = await job_service.get_unverified_jobs(limit=limit)
    return jobs


@router.get("/all")
async def get_all_jobs(
        job_service: JobVerifier,
        limit: int = Query(1000, gt=0, le=1000)
):
    """
    Get all jobs from Pinecone

    Args:
        job_service: JobVerifier service
        limit: The maximum number of jobs to return
    """
    jobs = await job_service.get_all_jobs(limit=limit)
    return jobs
