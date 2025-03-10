import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile, HTTPException


class S3Service:
    def __init__(self, bucket_name: str, client: boto3.client):
        """
        Initialize the S3 service with bucket name and region
        """
        self.bucket_name = bucket_name
        self.s3 = client

    async def upload_file(self, file: UploadFile, user_id: str, max_size_mb: int = 5) -> str:
        """
        Upload a file to S3 with user ownership metadata

        Args:
            file: The file to upload
            user_id: The ID of the user uploading the file
            max_size_mb: Maximum file size in MB

        Returns:
            The unique S3 key for the uploaded file

        Raises:
            HTTPException: If file upload fails or validation errors occur
        """
        # Create a unique filename that includes the user ID to enforce ownership
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_filename = f"resumes/{user_id}/{timestamp}-{uuid.uuid4()}.pdf"

        try:
            # Read file content
            file_content = await file.read()

            if len(file_content) > max_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds {max_size_mb}MB limit"
                )

            # Upload file to S3 with metadata to track ownership
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=file_content,
                ContentType='application/pdf',
                Metadata={
                    'user_id': user_id
                }
            )

            return unique_filename

        except ClientError as e:
            print(f"S3 upload error: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    def get_presigned_url(self, key: str, expiration_seconds: int = 3600) -> str:
        """
        Generate a presigned URL for accessing a file

        Args:
            key: The S3 key of the file
            expiration_seconds: URL expiration time in seconds

        Returns:
            Presigned URL for the file

        Raises:
            HTTPException: If URL generation fails
        """
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=expiration_seconds
            )
            return url
        except ClientError as e:
            print(f"S3 error details: {str(e)}")
            raise HTTPException(status_code=404, detail="File not found or access denied")
