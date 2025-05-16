import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from starlette.responses import JSONResponse

from dependencies import S3, Parser, PineconeClient, Embedder, CurrentUser
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.tools.extract_keywords import extract_keywords

router = APIRouter()


@router.post("/match")
async def calculate_resume_similarity(
        resume_parser: Parser,
        embedder: Embedder,
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
        resume_text = resume_parser.parse_pdf(file_bytes)

        matching_agent = ResumeMatchingAgent(
            embedder=embedder,
            resume_text=resume_text,
        )
        # Process using the matching agent
        result = await matching_agent.invoke(job_description, [])

        return result.answer

    except Exception as e:
        print(f"Error in resume matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_resume(
        s3_service: S3,
        parser: Parser,
        pinecone: PineconeClient,
        embedder: Embedder,
        current_user: CurrentUser,
        file: UploadFile = File(...),
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    user_id = current_user.user_id
    resume_index = pinecone.Index("resumes")

    # Read file bytes only once
    file_bytes = await file.read()

    # Reset file position for S3 upload
    file.file.seek(0)

    async def process_resume():
        try:
            # Parse text
            text = parser.parse_pdf(file_bytes)

            # Run text processing tasks concurrently
            keywords_task = extract_keywords(text)
            embeddings_task = embedder.get_embeddings([text])
            upload_task = s3_service.upload_file(
                file=file,
                user_id=user_id,
                content_type=file.content_type
            )

            keywords, embeddings, unique_filename = await asyncio.gather(
                keywords_task,
                embeddings_task,
                upload_task
            )

            return text, keywords, embeddings[0], unique_filename
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process resume: {str(e)}"
            )

    try:
        text, keywords, embedding, unique_filename = await process_resume()

        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "file_id": unique_filename,
            "keywords": keywords,
            "text": text,  # Store parsed text for future use
        }

        # Store in Pinecone
        resume_index.upsert(
            namespace="resumes",
            vectors=[{
                "id": user_id,
                "values": embedding.tolist(),
                "metadata": metadata,
            }]
        )

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_id": unique_filename,
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload resume: {str(e)}"
        )


@router.get("/view")
async def view_resume(
        s3_service: S3,
        key: str,
        current_user: CurrentUser,
):
    url = await s3_service.get_presigned_url(key)
    return JSONResponse(content={"url": url})


@router.delete("/delete")
async def delete_resume(
        s3_service: S3,
        pc: PineconeClient,
        key: str,
        current_user: CurrentUser,
):
    user_id = current_user.user_id  # Firebase UID

    # Security check: Ensure the user can only delete their own files
    if not key.startswith(f"resumes/{user_id}/"):
        raise HTTPException(status_code=403, detail="Not authorized to delete this file")

    try:
        # Delete file from S3
        s3_deleted = await s3_service.delete_file(key)

        # Delete vector from Pinecone
        resume_index = pc.Index("resumes")
        resume_index.delete(ids=[user_id], namespace="resumes")

        if not s3_deleted:
            # If S3 deletion failed but Pinecone succeeded, return partial success
            return JSONResponse(
                status_code=207,
                content={
                    "success": False,
                    "message": "File deleted from Pinecone but failed to delete from storage"
                }
            )

        return JSONResponse(content={"success": True, "message": "Resume deleted successfully"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete resume: {str(e)}")


@router.get("/keywords")
async def get_resume_keywords(
    pinecone: PineconeClient,
    current_user: CurrentUser,
):
    user_id = current_user.user_id
    
    resume_index = pinecone.Index("resumes")

    try:
        result = resume_index.fetch(ids=[user_id], namespace="resumes")
        vector = result.vectors.get(user_id)
        if not vector:
            raise HTTPException(status_code=404, detail="Resume not found in Pinecone")

        keywords = vector.metadata.get("keywords", [])
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone error: {str(e)}")