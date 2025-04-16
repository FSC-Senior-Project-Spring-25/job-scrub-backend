from typing import Dict, Any, Optional, List, Annotated

from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM, ResponseFormat


# Custom reducer function for merging dictionaries
def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries without overwriting existing keys"""
    result = dict1.copy()
    result.update(dict2)
    return result


class UserProfileState(TypedDict):
    # Input parameters
    user_id: str
    question: Optional[str]

    # Data gathered during execution
    resume_data: Optional[Dict[str, Any]]
    profile_info: Optional[Dict[str, Any]]

    # Analysis results (using custom dict_merger instead of operator.add)
    analysis_results: Annotated[Dict[str, Any], merge_dicts]

    # Final output
    answer: Optional[str]
    error: Optional[str]


class UserProfileAgent:
    """Agent for retrieving and analyzing user profile information using parallel processing"""

    def __init__(self, resumes_index: Pinecone.Index, llm: GeminiLLM):
        """
        Initialize the user profile agent with parallel processing capabilities

        Args:
            resumes_index: Pinecone index containing resume data
            llm: LLM service for profile analysis
        """
        self.resumes_index = resumes_index
        self.llm = llm
        self.graph = self._create_workflow()

    async def fetch_resume(self, state: UserProfileState) -> Dict[str, Any]:
        """Fetch resume data from Pinecone"""
        try:
            print(f"[USER_PROFILE_AGENT] Retrieving profile for user: {state['user_id']}")

            # Get resume data from Pinecone
            resume_data = await get_user_resume(self.resumes_index, state['user_id'])

            if not resume_data or not resume_data.get("text"):
                return {
                    "error": f"No resume data found for user {state['user_id']}",
                    "resume_data": None
                }

            return {
                "resume_data": resume_data
            }

        except Exception as e:
            print(f"[USER_PROFILE_AGENT] Error fetching resume: {str(e)}")
            return {
                "error": f"Failed to retrieve resume: {str(e)}",
                "resume_data": None
            }

    async def process_user_query(self, user_id: str, question: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query using the parallel processing graph

        Args:
            user_id: The ID of the user
            question: Optional specific question about the profile

        Returns:
            Dictionary with the response
        """
        initial_state = {
            "user_id": user_id,
            "question": question,
            "resume_data": None,
            "profile_info": None,
            "analysis_results": {},
            "answer": None,
            "error": None
        }

        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)

            # Return the final result
            return {
                "user_id": user_id,
                "answer": result.get("answer"),
                "error": result.get("error"),
                "profile": result.get("profile_info"),
                "analysis": result.get("analysis_results")
            }

        except Exception as e:
            print(f"[USER_PROFILE_AGENT] Graph execution error: {str(e)}")
            return {
                "user_id": user_id,
                "answer": f"Error processing request: {str(e)}",
                "error": str(e),
                "profile": None,
                "analysis": None
            }

    async def extract_profile(self, state: UserProfileState) -> Dict[str, Any]:
        """Extract structured profile information from resume text"""
        if state.get("error") or not state.get("resume_data"):
            return {"profile_info": None}

        resume_text = state["resume_data"].get("text", "")
        keywords = state["resume_data"].get("keywords", [])

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

        response = await self.llm.agenerate(
            system_prompt="You are an expert resume analyst tasked with extracting structured information from resume text.",
            user_message=prompt,
            response_format=ResponseFormat.JSON
        )

        if not response.success:
            return {
                "error": f"Failed to extract profile: {response.error}",
                "profile_info": None
            }

        return {
            "profile_info": response.content
        }

    async def analyze_skills(self, state: UserProfileState) -> Dict[str, Any]:
        """Analyze skills in parallel"""
        if state.get("error") or not state.get("profile_info"):
            return {"analysis_results": {}}

        profile = state["profile_info"]
        skills = profile.get("top_skills", [])

        # In a real implementation, you could do a deeper analysis here
        result = {
            "skills_analysis": {
                "count": len(skills),
                "technical_skills": [s for s in skills if self._is_technical_skill(s)],
                "soft_skills": [s for s in skills if self._is_soft_skill(s)],
                "skill_gaps": self._identify_skill_gaps(skills, profile.get("industry", ""))
            }
        }

        print(f"[USER_PROFILE_AGENT] Skills analysis completed")
        return {"analysis_results": result}

    async def analyze_experience(self, state: UserProfileState) -> Dict[str, Any]:
        """Analyze experience in parallel"""
        if state.get("error") or not state.get("profile_info"):
            return {"analysis_results": {}}

        profile = state["profile_info"]
        experience_years = profile.get("experience_years", 0)
        current_position = profile.get("current_position", {})

        result = {
            "experience_analysis": {
                "years": experience_years,
                "current_role": current_position.get("title"),
                "current_company": current_position.get("company"),
                "career_level": profile.get("career_level"),
                "experience_rating": self._rate_experience(experience_years, profile.get("career_level", ""))
            }
        }

        print(f"[USER_PROFILE_AGENT] Experience analysis completed")
        return {"analysis_results": result}

    async def analyze_education(self, state: UserProfileState) -> Dict[str, Any]:
        """Analyze education in parallel"""
        if state.get("error") or not state.get("profile_info"):
            return {"analysis_results": {}}

        profile = state["profile_info"]
        education = profile.get("education", {})
        certifications = profile.get("certifications", [])

        result = {
            "education_analysis": {
                "highest_degree": education.get("highest_degree"),
                "field": education.get("field"),
                "institution": education.get("institution"),
                "certifications_count": len(certifications),
                "education_relevance": self._assess_education_relevance(
                    education.get("field", ""),
                    profile.get("industry", "")
                )
            }
        }

        print(f"[USER_PROFILE_AGENT] Education analysis completed")
        return {"analysis_results": result}

    async def generate_answer(self, state: UserProfileState) -> Dict[str, Any]:
        """Generate final answer based on all parallel analyses"""
        if state.get("error"):
            return {"answer": f"Error: {state['error']}"}

        if not state.get("profile_info") or not state.get("resume_data"):
            return {"answer": "No profile information available."}

        # If there's a specific question, answer it
        if state.get("question"):
            return await self._answer_specific_question(state)

        # Otherwise, provide a general profile summary
        profile = state["profile_info"]
        analysis = state.get("analysis_results", {})

        # Create a comprehensive profile summary
        answer = f"Profile Summary for {profile.get('name', 'the user')}:\n\n"

        # Add current position
        if profile.get("current_position"):
            position = profile["current_position"]
            answer += f"Currently a {position.get('title', 'professional')} at {position.get('company', 'a company')}"
            if position.get("duration"):
                answer += f" for {position.get('duration')}"
            answer += ".\n\n"

        # Add experience
        if "experience_analysis" in analysis:
            exp = analysis["experience_analysis"]
            answer += f"Has {exp.get('years', 'several')} years of experience "
            answer += f"({exp.get('career_level', 'unknown')} level).\n\n"

        # Add skills
        if "skills_analysis" in analysis:
            skills = analysis["skills_analysis"]
            answer += f"Top skills include {', '.join(profile.get('top_skills', ['various skills']))}.\n"
            if skills.get("technical_skills"):
                answer += f"Technical strengths: {', '.join(skills['technical_skills'][:3])}.\n"
            if skills.get("skill_gaps"):
                answer += f"Potential skill development areas: {', '.join(skills['skill_gaps'][:2])}.\n\n"

        # Add education
        if "education_analysis" in analysis:
            edu = analysis["education_analysis"]
            if edu.get("highest_degree") and edu.get("field"):
                answer += f"Education: {edu.get('highest_degree')} in {edu.get('field')}"
                if edu.get("institution"):
                    answer += f" from {edu.get('institution')}"
                answer += ".\n"
            if profile.get("certifications") and len(profile["certifications"]) > 0:
                answer += f"Holds {len(profile['certifications'])} certifications including {', '.join(profile['certifications'][:2])}"
                if len(profile["certifications"]) > 2:
                    answer += " and others"
                answer += ".\n"

        return {"answer": answer}

    async def _answer_specific_question(self, state: UserProfileState) -> Dict[str, Any]:
        """Answer a specific question about the profile"""
        profile = state["profile_info"]
        resume_text = state["resume_data"]["text"]
        question = state["question"]
        analysis = state.get("analysis_results", {})

        prompt = f"""
        I need you to answer a question about a user's resume/profile.

        USER QUESTION: {question}

        USER RESUME:
        {resume_text}

        USER PROFILE SUMMARY:
        {profile}

        DETAILED ANALYSIS:
        {analysis}

        Answer the user's question specifically based on their resume data.
        Be precise and informative, citing specific details from their resume.
        If the information needed isn't available in their resume, acknowledge that
        and suggest what information they might want to add.
        """

        response = await self.llm.agenerate(
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
            "answer": response.content
        }

    # Helper methods for analyses
    def _is_technical_skill(self, skill: str) -> bool:
        """Determine if a skill is technical"""
        technical_keywords = [
            "python", "java", "javascript", "typescript", "react", "angular", "vue",
            "node", "aws", "azure", "gcp", "sql", "nosql", "mongodb", "database",
            "programming", "coding", "development", "software", "engineering", "machine learning",
            "ai", "data science", "devops", "cloud", "docker", "kubernetes"
        ]
        return any(keyword in skill.lower() for keyword in technical_keywords)

    def _is_soft_skill(self, skill: str) -> bool:
        """Determine if a skill is a soft skill"""
        soft_keywords = [
            "communication", "leadership", "teamwork", "collaboration", "problem solving",
            "critical thinking", "creativity", "time management", "organization", "adaptability",
            "flexibility", "interpersonal", "presentation", "public speaking", "writing",
            "negotiation", "conflict resolution", "emotional intelligence", "empathy"
        ]
        return any(keyword in skill.lower() for keyword in soft_keywords)

    def _identify_skill_gaps(self, skills: List[str], industry: str) -> List[str]:
        """Identify potential skill gaps based on industry"""
        # This is a simplified implementation
        all_skills_lower = [s.lower() for s in skills]
        gaps = []

        industry_skills = {
            "technology": ["cloud", "security", "devops", "agile", "microservices"],
            "finance": ["financial analysis", "risk assessment", "regulatory compliance"],
            "healthcare": ["healthcare compliance", "patient care", "medical terminology"],
            "marketing": ["digital marketing", "seo", "content strategy", "analytics"]
        }

        if industry.lower() in industry_skills:
            for skill in industry_skills[industry.lower()]:
                if not any(skill in s for s in all_skills_lower):
                    gaps.append(skill)

        return gaps[:3]  # Return top 3 gaps

    def _rate_experience(self, years: int, career_level: str) -> str:
        """Rate experience level"""
        if years < 2:
            return "Entry-level"
        elif years < 5:
            return "Mid-level"
        elif years < 10:
            return "Senior"
        else:
            return "Expert"

    def _assess_education_relevance(self, field: str, industry: str) -> str:
        """Assess education relevance to industry"""
        if not field or not industry:
            return "Unknown"

        field_lower = field.lower()
        industry_lower = industry.lower()

        # Direct matches
        if field_lower in industry_lower or industry_lower in field_lower:
            return "Highly relevant"

        # Related fields
        tech_fields = ["computer science", "software engineering", "information technology"]
        business_fields = ["business", "finance", "economics", "marketing"]

        if industry_lower in ["technology", "software"] and any(tf in field_lower for tf in tech_fields):
            return "Highly relevant"
        if industry_lower in ["finance", "banking", "business"] and any(bf in field_lower for bf in business_fields):
            return "Highly relevant"

        return "Somewhat relevant"

    def _create_workflow(self) -> CompiledStateGraph:
        """Build the processing graph for parallel execution"""
        builder = StateGraph(UserProfileState)

        # Add nodes
        builder.add_node("fetch_resume", self.fetch_resume)
        builder.add_node("extract_profile", self.extract_profile)
        builder.add_node("analyze_skills", self.analyze_skills)
        builder.add_node("analyze_experience", self.analyze_experience)
        builder.add_node("analyze_education", self.analyze_education)
        builder.add_node("generate_answer", self.generate_answer)

        # Define the graph flow
        builder.add_edge(START, "fetch_resume")
        builder.add_edge("fetch_resume", "extract_profile")

        # Fan out for parallel analysis
        builder.add_edge("extract_profile", "analyze_skills")
        builder.add_edge("extract_profile", "analyze_experience")
        builder.add_edge("extract_profile", "analyze_education")

        # Fan in to generate final answer
        builder.add_edge("analyze_skills", "generate_answer")
        builder.add_edge("analyze_experience", "generate_answer")
        builder.add_edge("analyze_education", "generate_answer")

        builder.add_edge("generate_answer", END)

        workflow = builder.compile()
        print("User Profile Workflow:")
        print(workflow.get_graph().draw_ascii())

        return workflow
