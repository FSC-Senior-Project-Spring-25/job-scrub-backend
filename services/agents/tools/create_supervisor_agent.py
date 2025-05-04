from typing import List, Optional, Dict, Any

from fastapi import UploadFile
from pinecone import Pinecone

from models.chat import Message
from services.agents.job_search_agent import JobSearchAgent
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.supervisor_agent import SupervisorAgent
from services.agents.tools.get_user_resume import get_user_resume
from services.agents.user_profile_agent import UserProfileAgent
from services.agents.user_search_agent import UserSearchAgent
from services.llm.base.llm import LLM
from services.llm.groq import GroqLLM
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder


async def create_supervisor_agent(
        user_id: str,
        pinecone_client: Pinecone,
        llm: LLM = GroqLLM(),
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        resume_file: Optional[UploadFile] = None,
        resume_parser: ResumeParser = ResumeParser(),
        text_embedder: TextEmbedder = TextEmbedder(),
) -> SupervisorAgent:
    """
    Create a new SupervisorAgent instance for a chat session

    Args:
        user_id: Current user's ID
        pinecone_client: Pinecone client instance
        llm: LLM instance
        conversation_history: Previous conversation history
        resume_file: Optional resume file upload
        resume_parser: Resume parser instance
        text_embedder: Text embedder instance

    Returns:
        A new SupervisorAgent instance with initialized state
    """
    # Process resume file if provided
    processed_file, resume_file_data = await process_resume_file(resume_file, resume_parser)

    # Get resume data either from uploaded file or from database
    resume_data = await get_resume_data(
        user_id=user_id,
        pinecone_client=pinecone_client,
        resume_file_data=resume_file_data,
        text_embedder=text_embedder
    )

    # Convert conversation history to Message objects
    history = convert_conversation_history(conversation_history)

    # Create all agent instances
    agents = create_agent_instances(
        pinecone_client=pinecone_client,
        resume_data=resume_data,
        text_embedder=text_embedder,
        llm=llm
    )

    # Create and return the supervisor agent
    supervisor = SupervisorAgent(
        llm=llm,
        pc=pinecone_client,
        resume_matcher=agents["resume_matcher"],
        resume_enhancer=agents["resume_enhancer"],
        user_profile_agent=agents["user_profile"],
        job_search_agent=agents["job_search"],
        user_search_agent=agents["user_search"],
        resume_data=resume_data,
        processed_conversation_history=history,
        processed_files=[processed_file] if processed_file else None,
    )

    return supervisor


async def process_resume_file(
        resume_file: Optional[UploadFile],
        resume_parser: ResumeParser
) -> tuple[Optional[dict], Optional[dict]]:
    """Process uploaded resume file if provided"""
    processed_file = None
    resume_file_data = None

    if resume_file:
        file_bytes = await resume_file.read()

        if resume_file.filename.endswith('.pdf'):
            content = resume_parser.parse_pdf(file_bytes)
            resume_file_data = {
                "text": content,
                "filename": resume_file.filename
            }

            processed_file = {
                "filename": resume_file.filename,
                "type": "pdf",
                "bytes": file_bytes,
                "content": content
            }
            print(f"[CREATE_SUPERVISOR] Processed resume file: {resume_file.filename}")

    return processed_file, resume_file_data


async def get_resume_data(
        user_id: str,
        pinecone_client: Pinecone,
        resume_file_data: Optional[dict],
        text_embedder: TextEmbedder
) -> dict:
    """Get resume data either from uploaded file or database"""
    if resume_file_data:
        # Generate embedding for the uploaded resume
        print(f"[CREATE_SUPERVISOR] Using uploaded resume: {resume_file_data['filename']}")
        resume_text = resume_file_data["text"]
        embeddings = await text_embedder.get_embeddings([resume_text])
        resume_vector = embeddings[0].tolist()
        print(f"[CREATE_SUPERVISOR] Generated vector for uploaded resume")
        print(f"[CREATE_SUPERVISOR] Resume text: {resume_text[:50]}...")
        print(f"[CREATE_SUPERVISOR] Resume vector: {resume_vector}")
        return {
            "text": resume_text,
            "vector": resume_vector,
            "source": "uploaded_file"
        }
    else:
        # Fetch resume from database
        print(f"[CREATE_SUPERVISOR] Fetching resume for user: {user_id}")
        try:
            resume_data = await get_user_resume(
                index=pinecone_client.Index("resumes"),
                user_id=user_id
            )
            print(f"[CREATE_SUPERVISOR] Resume data retrieved")
            return resume_data
        except ValueError as e:
            # User doesn't have a resume uploaded, create a default response
            print(f"[CREATE_SUPERVISOR] No resume found for user: {user_id}")
            # Create a default embedding (zero vector)
            default_text = "No resume available. Please upload a resume to get personalized assistance."
            default_embeddings = [0] * text_embedder.dim

            return {
                "text": default_text,
                "vector": default_embeddings,
                "source": "default",
                "file_id": None,
                "filename": None,
                "keywords": []
            }


def convert_conversation_history(conversation_history: Optional[List[Dict[str, Any]]]) -> List[Message]:
    """Convert raw conversation history to Message objects"""
    history = []
    if conversation_history:
        history = [
            Message(
                role=item["role"],
                content=item["content"],
                timestamp=item["timestamp"],
                metadata=item.get("metadata", {})
            )
            for item in conversation_history
        ]
    return history


def create_agent_instances(
        pinecone_client: Pinecone,
        resume_data: dict,
        text_embedder: TextEmbedder,
        llm: LLM,
) -> dict:
    """Create all required agent instances"""
    return {
        "resume_matcher": ResumeMatchingAgent(
            embedder=text_embedder,
            resume_text=resume_data.get("text"),
        ),
        "resume_enhancer": ResumeEnhancementAgent(
            resume_text=resume_data.get("text"),
            llm=llm
        ),
        "user_profile": UserProfileAgent(
            resume_text=resume_data.get("text"),
            llm=llm
        ),
        "job_search": JobSearchAgent(
            llm=llm,
            job_index=pinecone_client.Index("job-postings"),
            resume_vector=resume_data["vector"]
        ),
        "user_search": UserSearchAgent(
            llm=llm,
            resume_index=pinecone_client.Index("resumes"),
            resume_vector=resume_data["vector"]
        )
    }
