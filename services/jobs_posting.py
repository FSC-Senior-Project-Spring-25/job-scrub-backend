from pinecone import Pinecone

from models.job_report import JobReport
from services.text_embedder import TextEmbedder


class JobPostingService:
    def __init__(self, embedder: TextEmbedder, index: Pinecone.Index):
        self.embedder = embedder
        self.index = index

    def create_job_embedding(self, job: JobReport) -> dict:
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

        return {
            "id": f"job_{abs(hash(job.url))}",  # Create unique ID from link
            "values": embedding.tolist(),
            "metadata": metadata,
        }

    async def post_job(self, job: JobReport) -> str:
        """
        Upserts a job to Pinecone after creating an embedding, metadata, and ID

        :return: the ID of the job
        """
        # Create embedding
        embedding = self.create_job_embedding(job)

        # Upsert to Pinecone
        self.index.upsert(vectors=[embedding])

        return embedding["id"]