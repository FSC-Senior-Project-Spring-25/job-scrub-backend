import asyncio
import uuid

import aiohttp
from async_lru import alru_cache
from pinecone import Pinecone

from models.job_report import JobReport
from services.text_embedder import TextEmbedder


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

    @alru_cache(maxsize=256)
    async def get_coordinates(self, location: str) -> tuple[float, float]:
        """
        Get the coordinates of a location (city, state) using OSM with caching
        :return: the coordinates as a tuple of (lat, lon)
        """
        if not location:
            return 0.0, 0.0

        try:
            async with self.session.get(
                    url="https://nominatim.openstreetmap.org/search",
                    params={
                        "format": "json",
                        "q": location
                    },
                    headers={
                        "User-Agent": "JobPostingService/1.0"
                    }
            ) as response:
                if response.status != 200:
                    return 0.0, 0.0

                data = await response.json()
                if data:
                    return float(data[0]["lat"]), float(data[0]["lon"])

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, IndexError, KeyError):
            print(f"Error getting coordinates for {location}")

        return 0.0, 0.0


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

        lat, lon = await self.get_coordinates(job.location)
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
        # Create embedding
        embedding = await self.create_job_posting(job)
        # Upsert to Pinecone
        self.index.upsert(
            namespace="jobs",
            vectors=[embedding]
        )

        return embedding["id"]
