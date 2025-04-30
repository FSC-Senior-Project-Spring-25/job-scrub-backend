from typing import List, Optional, Dict, Any

from pinecone import Pinecone

from models.chat import Message
from services.agents.job_search_agent import JobSearchAgent
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.supervisor_agent import SupervisorAgent
from services.agents.tools.get_user_resume import get_user_resume
from services.agents.user_profile_agent import UserProfileAgent
from services.agents.user_search_agent import UserSearchAgent
from services.gemini import GeminiLLM
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder


async def create_supervisor_agent(
        user_id: str,
        pinecone_client: Pinecone,
        llm: GeminiLLM = GeminiLLM(),
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        resume_parser: ResumeParser = ResumeParser(),
        text_embedder: TextEmbedder = TextEmbedder(),
) -> SupervisorAgent:
    """
    Create a new SupervisorAgent instance for a chat session

    Args:
        message: The current message being processed
        conversation_history: Previous conversation history
        user_id: Current user's ID
        pinecone_client: Pinecone client instance
        llm: LLM instance
        resume_parser: Resume parser instance
        text_embedder: Text embedder instance
        files: Optional list of processed files

    Returns:
        A new SupervisorAgent instance with initialized state
    """
    # Convert conversation history to Message objects
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

    # Pre-fetch the user's resume text
    resume_data = await get_user_resume(
        index=pinecone_client.Index("resumes"),
        user_id=user_id
    )
    resume_text = resume_data.get("text", "")
    print("Resume Data:", resume_data)

    # Create the matcher agent
    resume_matcher = ResumeMatchingAgent(
        resume_parser=resume_parser,
        text_embedder=text_embedder,
        llm=llm
    )

    # Create enhancer agent
    resume_enhancer = ResumeEnhancementAgent(llm=llm)

    # Create user profile agent
    user_profile_agent = UserProfileAgent(llm=llm)

    # Create job search agent with user's resume vector
    job_search_agent = JobSearchAgent(
        llm=llm,
        job_index=pinecone_client.Index("job-postings"),
        resume_vector=resume_data["vector"]
    )

    # Create user search agent
    user_search_agent = UserSearchAgent(
        llm=llm,
        resume_index=pinecone_client.Index("resumes"),
        resume_vector=resume_data["vector"]
    )

    # Create and return the supervisor agent
    supervisor = SupervisorAgent(
        llm=llm,
        pc=pinecone_client,
        resume_matcher=resume_matcher,
        resume_enhancer=resume_enhancer,
        user_profile_agent=user_profile_agent,
        job_search_agent=job_search_agent,
        user_search_agent=user_search_agent,
        resume_data=resume_data,
        processed_conversation_history=history
    )

    return supervisor