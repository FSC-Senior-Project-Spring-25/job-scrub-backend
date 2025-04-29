from fastapi import APIRouter
from fastapi.params import Query

from dependencies import JobPoster, JobVerifier
from models.job_report import JobReport

router = APIRouter()


@router.post("/report")
async def create_job_report(report: JobReport, job_service: JobPoster):
    id = await job_service.post_job(report)
    return {"message": "Job report created successfully with ID: " + id}


@router.patch("/verify/{job_id}")
async def verify_job(job_id: str, verified: bool, report: JobReport, job_service: JobVerifier):
    await job_service.verify_job(job_id, verified, report)
    return {"message": "Job verified successfully"}


@router.delete("/delete/{job_id}")
async def delete_job(job_id: str, job_service: JobVerifier):
    await job_service.delete_job(job_id)
    return {"message": "Job deleted successfully"}


@router.get("/unverified")
async def get_unverified_jobs(
        job_service: JobVerifier,
        limit: int = Query(1000, gt=0, le=1000)
):
    jobs = await job_service.get_unverified_jobs(limit=limit)
    return jobs

@router.get("/all")
async def get_all_jobs(
        job_service: JobVerifier,
        limit: int = Query(1000, gt=0, le=1000)
):
    jobs = await job_service.get_all_jobs(limit=limit)
    return jobs
