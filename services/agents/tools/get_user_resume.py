from typing import Dict, Any

from pinecone import Pinecone


async def get_user_resume(index: Pinecone.Index, user_id: str) -> Dict[str, Any]:
    """Retrieve user resume data from Pinecone"""
    # Query Pinecone for user's resume data
    try:
        # Fetch the user's resume vector from Pinecone
        fetched_resume_data = index.fetch(
            ids=[user_id],
            namespace="resumes",
        )

        # Check if we have any vectors returned
        if not fetched_resume_data.vectors or user_id not in fetched_resume_data.vectors:
            raise ValueError(f"No resume data found for user {user_id}")

        # Access the vector data correctly from the response structure
        vector_data = fetched_resume_data.vectors[user_id]

        # Get metadata from the vector
        if not vector_data.metadata:
            raise ValueError(f"No metadata found in resume vector for user {user_id}")
        print(vector_data)
        return {
            "file_id": vector_data.metadata.get("file_id"),
            "filename": vector_data.metadata.get("filename"),
            "keywords": vector_data.metadata.get("keywords", []),
            "text": vector_data.metadata.get("text", ""),
            "vector": vector_data.values,
        }
    except Exception as e:
        raise ValueError(f"Failed to retrieve resume data: {str(e)}")
