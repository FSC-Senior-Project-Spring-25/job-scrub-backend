import asyncio
import os
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM, ResponseFormat

load_dotenv()

class ProfileAgentState(MessagesState):
    """
    The agent state for user profile analysis.
    Holds the user_id, prompt, resume text, and conversation messages.
    """
    prompt: Optional[str]
    resume_text: Optional[str]


class UserProfileAgent:
    """Agent for retrieving and analyzing user profile information using ReAct pattern"""

    def __init__(self, llm: GeminiLLM):
        """
        Initialize the user profile agent with ReAct capabilities

        Args:
            llm: LLM service for profile analysis
        """
        self.llm = llm

        # Create the tools
        self.tools = self._create_tools()

        # Bind the tools to the LLM
        self.llm_with_tools = self.llm.chat.bind_tools(self.tools)

        # Build the ReAct agent state graph
        self.builder = StateGraph(ProfileAgentState)
        self.builder.add_node("think", self.think)
        self.builder.add_node("tools", ToolNode(self.tools))
        self.builder.add_edge(START, "think")
        self.builder.add_conditional_edges("think", tools_condition)
        self.builder.add_edge("tools", "think")
        self.agent = self.builder.compile()

        print("==" * 20)
        print("User Profile Agent Graph:")
        print("==" * 20)
        self.agent.get_graph().print_ascii()

    def _create_tools(self):
        """
        Creates and returns all the tools for the UserProfileAgent.

        Returns:
            List of tools for the ReAct agent.
        """
        tools = [
            self._create_extract_profile_tool(),
        ]
        return tools

    def _create_extract_profile_tool(self):
        @tool(parse_docstring=True)
        async def extract_profile_tool(resume_text: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Extract structured profile information from resume text.

            Args:
                resume_text: The complete resume text
                keywords: Optional list of keywords extracted from the resume

            Returns:
                A dictionary containing structured profile information.
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

            response = await self.llm.agenerate(
                system_prompt="You are an expert resume analyst tasked with extracting structured information from resume text.",
                user_message=prompt,
                response_format=ResponseFormat.JSON
            )

            if not response.success:
                return {
                    "success": False,
                    "error": f"Failed to extract profile: {response.error}",
                    "profile_info": None
                }

            return {
                "success": True,
                "profile_info": response.content
            }

        return extract_profile_tool

    def think(self, state: ProfileAgentState) -> Dict[str, Any]:
        """
        Assistant node that evaluates the current state and executes the next required tool.
        """
        prompt = state.get("prompt", "")
        resume_text = state.get("resume_text", "")

        sys_msg = SystemMessage(
            content=(
                "You are a user profile analysis assistant that helps analyze resume data "
                "and answer questions about user profiles. Follow this process:\n"
                "1. Then extract structured profile information using extract_profile_tool\n"
                "2. Finally, answer the user's question or provide a profile summary using profile information\n\n"
            )
        )

        messages: list[BaseMessage] = [sys_msg] + state["messages"]

        # For the first message, add a human message that initiates the process
        if len(state["messages"]) == 1:
            messages.append(
                HumanMessage(
                    content=(
                        f"Please answer {prompt}"
                        f"Given the resume text: {resume_text}, "
                    )
                )
            )

        invocation = self.llm_with_tools.invoke(messages)
        return {"messages": state["messages"] + [invocation]}

    async def process_user_query(self, resume_text: str, prompt: str) -> Dict[str, Any]:
        """
        Process a query about a resume

        Args:
            resume_text: The resume text to analyze
            prompt: The question or prompt to answer related to the resume

        Returns:
            Dictionary with the response
        """
        initial_state = {
            "resume_text": resume_text,
            "question": prompt,
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this resume and answer: {prompt}"
                }
            ]
        }

        result = await self.agent.ainvoke(initial_state)
        final_message = result["messages"][-1].content if result["messages"] else ""
        print("Final User Profile Message:", final_message)
        return {
            "answer": final_message,
            "error": None
        }


if __name__ == "__main__":
    # Example usage
    user_id = "oPmOJhSE0VQid56yYyg19hdH5DV2"
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index("resumes")
    resume_data = asyncio.run(get_user_resume(index, user_id))
    if not resume_data.get("text"):
        print("Resume text not found")
        exit(1)

    # Instantiate GeminiLLM
    llm = GeminiLLM()
    user_profile_agent = UserProfileAgent(llm)

    resume_text = resume_data["text"]
    prompt = "What are the key skills and experience of this candidate?"

    response = asyncio.run(user_profile_agent.process_user_query(resume_text, prompt))
    print(response)