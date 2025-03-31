from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from starlette.responses import JSONResponse

from dependencies import MatchingAgent, S3, get_current_user

router = APIRouter()


@router.post("/match")
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


@router.post("/upload")
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


@router.get("/view")
async def view_resume(
        s3_service: S3,
        key: str,
        current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]  # Firebase UID

    # Security check: Ensure the user can only access their own files
    if not key.startswith(f"resumes/{user_id}/"):
        raise HTTPException(status_code=403, detail="Not authorized to access this file")

    url = await s3_service.get_presigned_url(key)
    return JSONResponse(content={"url": url})


@router.delete("/delete")
async def delete_resume(
        s3_service: S3,
        key: str,
        current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]  # Firebase UID

    # Security check: Ensure the user can only delete their own files
    if not key.startswith(f"resumes/{user_id}/"):
        raise HTTPException(status_code=403, detail="Not authorized to delete this file")

    if await s3_service.delete_file(key):
        return JSONResponse(content={"success": True})

    # Must have failed to delete
    raise HTTPException(status_code=500, detail="Failed to delete file")