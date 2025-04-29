import aiohttp
from fastapi import HTTPException
from pinecone import Pinecone

from models.job_report import JobReport
from services.text_embedder import TextEmbedder
from utils.coordinates import get_coordinates


class JobsVerificationService:

    def __init__(self, session: aiohttp.ClientSession, index: Pinecone.Index, embedder: TextEmbedder):
        self.session = session
        self.index = index
        self.embedder = embedder

    async def get_unverified_jobs(self, limit=1000) -> list[dict]:
        """
        Get unverified jobs from the Pinecone index using a vector query with filter

        Args:
            limit: The maximum number of jobs to return

        Returns:
            A list of unverified job postings
        """
        try:
            # Create a dummy vector of the proper dimension for the query
            dummy_vector = [0.0] * self.embedder.dim

            # Query the index with the dummy vector but filter for unverified jobs
            response = self.index.query(
                namespace="jobs",
                vector=dummy_vector,
                filter={"verified": False},
                top_k=limit,
                include_metadata=True
            )

            # Format the results
            unverified_jobs = []
            for match in response.matches:
                unverified_jobs.append({
                    "id": match.id,
                    "metadata": match.metadata,
                })

            return unverified_jobs
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def verify_job(self, job_id: str, verified: bool, job_report: JobReport) -> None:
        """
        Verify a job in the Pinecone index and update its metadata with the provided JobReport.
        Regenerates embeddings if title or description have been modified.

        Args:
            job_id: The ID of the job to verify
            verified: Whether the job is verified or not
            job_report: Optional JobReport containing updated job information
        """
        # Get current job data
        current_data = self.index.fetch(ids=[job_id], namespace="jobs").vectors.get(job_id)
        if not current_data:
            raise HTTPException(status_code=404, detail="Job not found")

        try:
            current_metadata = current_data.metadata

            # Start with base metadata update
            final_metadata = {"verified": verified}

            # Convert JobReport to dict and update final metadata
            amended_metadata = job_report.model_dump(exclude_none=True, by_alias=True, exclude={"location"})
            amended_metadata["lat"] = job_report.location.lat
            amended_metadata["lon"] = job_report.location.lon
            amended_metadata["address"] = job_report.location.address

            # Convert enum to string
            amended_metadata["jobType"] = job_report.job_type.value
            amended_metadata["locationType"] = job_report.location_type.value

            final_metadata.update(amended_metadata)

            # Check if title or description changed (requiring vector update)
            title_changed = amended_metadata.get('title') != current_metadata.get('title')
            desc_changed = amended_metadata.get('description') != current_metadata.get('description')

            if title_changed or desc_changed:
                # Generate new embedding using the updated content
                combined_text = f"{final_metadata['title']} {final_metadata['description']}"
                new_embedding = await self.embedder.get_embeddings([combined_text])

                # Update both vector and metadata
                self.index.upsert(
                    namespace="jobs",
                    vectors=[{
                        "id": job_id,
                        "values": new_embedding[0].tolist(),
                        "metadata": final_metadata
                    }]
                )
                return

            # If no embedding update needed, just update metadata
            self.index.update(
                namespace="jobs",
                id=job_id,
                set_metadata=final_metadata
            )
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail="Failed to verify job: " + str(e))

    async def delete_job(self, job_id: str) -> None:
        """
        Delete a job from the Pinecone index.

        Args:
            job_id: The ID of the job to delete
        """
        # Check if job exists
        job = self.index.fetch(ids=[job_id], namespace="jobs")
        if not job.vectors:
            raise HTTPException(status_code=404, detail="Job not found")

        self.index.delete(ids=[job_id], namespace="jobs")
