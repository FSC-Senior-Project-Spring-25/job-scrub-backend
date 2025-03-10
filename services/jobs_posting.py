import uuid

import aiohttp
from fastapi import HTTPException
from pinecone import Pinecone

from models.job_report import JobReport
from services.text_embedder import TextEmbedder
from utils.coordinates import get_coordinates


class JobsPostingService:
    def __init__(
            self,
            embedder: TextEmbedder,
            index: Pinecone.Index,
            session: aiohttp.ClientSession,
    ):
        self.embedder = embedder
        self.index = index
        self.session = session


    async def create_job_posting(self, job: JobReport) -> dict:
        """
        Creates the embedding, metadata, and ID for a job, prioritizing the title and description
        in the embedding

        :param job: the job to post with a mandatory title, company, and URL
        :return: the id, embedding, and metadata as a dictionary
        """
        # Combine title and description for a richer embedding
        combined_text = f"{job.title} {job.description}"

        # Get embedding
        embedding = self.embedder.get_embeddings([combined_text])[0]

        # Prepare metadata
        metadata = job.model_dump(exclude_none=True, by_alias=True)

        lat, lon = await get_coordinates(session=self.session, location=job.location)
        metadata["lat"] = lat
        metadata["lon"] = lon

        # All jobs are unverified by default
        metadata["verified"] = False
        return {
            "id": f"job_{uuid.uuid4()}",  # Generate unique ID using UUID
            "values": embedding.tolist(),
            "metadata": metadata,
        }


    async def post_job(self, job: JobReport) -> str:
        """
        Upserts a job to Pinecone after creating an embedding, metadata, and ID

        :return: the ID of the job
        """
        try:
            # Create embedding
            embedding = await self.create_job_posting(job)
            # Upsert to Pinecone
            self.index.upsert(
                namespace="jobs",
                vectors=[embedding]
            )

            return embedding["id"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))