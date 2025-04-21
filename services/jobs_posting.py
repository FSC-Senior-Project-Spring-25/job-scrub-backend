import uuid
import aiohttp
import asyncio
from fastapi import HTTPException
from pinecone import Pinecone

from models.job_report import JobReport
from services.agents.tools.extract_keywords import extract_keywords
from services.gemini import GeminiLLM
from services.text_embedder import TextEmbedder


class JobsPostingService:
    def __init__(
            self,
            embedder: TextEmbedder,
            index: Pinecone.Index,
            llm: GeminiLLM,
            session: aiohttp.ClientSession,
    ):
        self.embedder = embedder
        self.index = index
        self.llm = llm
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

        # Run tasks concurrently
        embedding_task = self.embedder.get_embeddings([combined_text])
        keywords_task = extract_keywords(combined_text, self.llm)

        embedding, keywords = await asyncio.gather(embedding_task, keywords_task)

        # Prepare metadata
        metadata = job.model_dump(exclude_none=True, by_alias=True, exclude={"location"})
        metadata["lat"] = job.location.lat
        metadata["lon"] = job.location.lon
        metadata["address"] = job.location.address
        metadata["keywords"] = keywords
        # Convert enum to string
        metadata["jobType"] = job.job_type.value
        metadata["locationType"] = job.location_type.value

        # All jobs are unverified by default
        metadata["verified"] = False
        return {
            "id": f"job_{uuid.uuid4()}",  # Generate unique ID using UUID
            "values": embedding[0].tolist(),
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