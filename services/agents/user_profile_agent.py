from typing import Dict, Any, Optional, List

from pinecone import Pinecone

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM, ResponseFormat


class UserProfileAgent:
    """Agent for retrieving and analyzing user profile information"""

    def __init__(self, resumes_index: Pinecone.Index, llm: GeminiLLM):
        """
        Initialize the user profile agent

        Args:
            resumes_index: Pinecone index containing resume data
            llm: LLM service for profile analysis
        """
        self.resumes_index = resumes_index
        self.llm = llm

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve and analyze user profile data from their resume

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary with user profile information
        """
        try:
            print(f"[USER_PROFILE_AGENT] Retrieving profile for user: {user_id}")

            # Get resume data from Pinecone
            resume_data = await get_user_resume(self.resumes_index, user_id)

            if not resume_data or not resume_data.get("text"):
                return {
                    "error": "No resume data found for user",
                    "user_id": user_id,
                    "profile": None
                }

            # Extract key profile information from resume text
            profile = await self._extract_profile_info(resume_data["text"], resume_data.get("keywords", []))

            print(f"[USER_PROFILE_AGENT] Profile extracted for user: {user_id}")

            return {
                "user_id": user_id,
                "profile": profile,
                "keywords": resume_data.get("keywords", []),
                "file_id": resume_data.get("file_id"),
                "resume_text": resume_data.get("text", "")
            }

        except Exception as e:
            print(f"[USER_PROFILE_AGENT] Error: {str(e)}")
            return {
                "error": str(e),
                "user_id": user_id,
                "profile": None
            }

    async def analyze_user_question(self, user_id: str, question: str) -> Dict[str, Any]:
        """
        Analyze a specific user question about their profile or resume

        Args:
            user_id: The ID of the user
            question: The specific question about their profile/resume

        Returns:
            Answer to the user's question using profile data
        """
        try:
            # Get the user's profile data first
            user_data = await self.get_user_profile(user_id)

            if "error" in user_data and user_data["profile"] is None:
                return {
                    "answer": f"I couldn't find your resume data. {user_data['error']}",
                    "error": user_data["error"]
                }

            # Use LLM to answer the specific question
            profile_data = user_data["profile"]
            resume_text = user_data["resume_text"]

            prompt = f"""
            I need you to answer a question about a user's resume/profile.

            USER QUESTION: {question}

            USER RESUME:
            {resume_text}

            USER PROFILE SUMMARY:
            {profile_data}

            Answer the user's question specifically based on their resume data.
            Be precise and informative, citing specific details from their resume.
            If the information needed isn't available in their resume, acknowledge that
            and suggest what information they might want to add.
            """

            response = await self.llm.generate(
                system_prompt="You are a helpful assistant analyzing a user's resume data to answer their questions.",
                user_message=prompt,
                response_format=ResponseFormat.RAW
            )

            if not response.success:
                return {
                    "answer": "I encountered an error analyzing your resume data.",
                    "error": response.error
                }

            return {
                "answer": response.content,
                "profile": profile_data
            }

        except Exception as e:
            print(f"[USER_PROFILE_AGENT] Error analyzing question: {str(e)}")
            return {
                "answer": "I encountered an error while analyzing your question.",
                "error": str(e)
            }

    async def _extract_profile_info(self, resume_text: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Extract structured profile information from resume text

        Args:
            resume_text: The full text of the resume
            keywords: Keywords extracted from the resume

        Returns:
            Structured profile information
        """
        prompt = f"""
        Extract key professional profile information from the following resume text.
        Focus on:

        1. Name and contact information
        2. Current or most recent job title and company
        3. Years of experience in their field
        4. Top skills (based on both explicit mentions and implicit evidence)
        5. Education level and field
        6. Industry specialization
        7. Career level (entry, mid, senior, executive)
        8. Key projects mentioned with brief descriptions
        9. Certifications and qualifications

        Resume text:
        {resume_text}

        Resume keywords: {", ".join(keywords) if keywords else "No keywords available"}

        Return the information as a JSON object with the following structure:
        {{
            "name": "Full Name",
            "contact": {{
                "email": "email@example.com",
                "phone": "optional phone number",
                "location": "City, State"
            }},
            "current_position": {{
                "title": "Job Title",
                "company": "Company Name",
                "duration": "Duration in role (e.g., '2 years')"
            }},
            "experience_years": 5,
            "top_skills": ["Skill 1", "Skill 2", "Skill 3"],
            "education": {{
                "highest_degree": "Degree Type",
                "field": "Field of Study",
                "institution": "School Name"
            }},
            "industry": "Primary industry",
            "career_level": "entry/mid/senior/executive",
            "projects": [
                {{
                    "name": "Project Name",
                    "description": "Brief description",
                    "technologies": ["Tech 1", "Tech 2"]
                }}
            ],
            "certifications": ["Certification 1", "Certification 2"]
        }}

        If certain information is not available, use null for that field.
        """

        response = await self.llm.generate(
            system_prompt="You are an expert resume analyst tasked with extracting structured information from resume text.",
            user_message=prompt,
            response_format=ResponseFormat.JSON
        )

        if not response.success:
            return {
                "error": f"Failed to extract profile: {response.error}",
                "basic_info": {
                    "keywords": keywords
                }
            }

        return response.content

    async def get_user_resume_data(self, user_id):
        """Retrieve user resume data from Pinecone"""
        # Query Pinecone for user's resume data
        try:
            # Fetch the user's resume vector from Pinecone
            fetched_resume_data = self.resumes_index.fetch(
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

            return {
                "file_id": vector_data.metadata.get("file_id"),
                "filename": vector_data.metadata.get("filename"),
                "keywords": vector_data.metadata.get("keywords", []),
                "text": vector_data.metadata.get("text", "")
            }
        except Exception as e:
            raise ValueError(f"Failed to retrieve resume data: {str(e)}")