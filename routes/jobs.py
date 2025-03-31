from fastapi import APIRouter

from models.job_report import JobReport
from dependencies import JobPostingService, JobVerificationService

router = APIRouter()


@router.post("/report")
async def create_job_report(report: JobReport, job_service: JobPostingService):
    id = await job_service.post_job(report)
    return {"message": "Job report created successfully with ID: " + id}


@router.patch("/verify/{job_id}")
async def verify_job(job_id: str, verified: bool, report: JobReport, job_service: JobVerificationService):
    await job_service.verify_job(job_id, verified, report)
    return {"message": "Job verified successfully"}


@router.delete("/delete/{job_id}")
async def delete_job(job_id: str, job_service: JobVerificationService):
    await job_service.delete_job(job_id)
    return {"message": "Job deleted successfully"}


@router.get("/unverified")
async def get_unverified_jobs(job_service: JobVerificationService):
    jobs = await job_service.get_unverified_jobs()
    return jobs