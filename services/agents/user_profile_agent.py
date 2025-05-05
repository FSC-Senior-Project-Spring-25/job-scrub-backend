import json
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from services.agents.base.agent import ReActAgent, AgentResponse
from services.llm.base.llm import LLM
from services.llm.groq import GroqLLM

load_dotenv()


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class CurrentPosition(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None


class Education(BaseModel):
    highest_degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)


class ProfileInfoModel(BaseModel):
    name: Optional[str] = None
    contact: ContactInfo = ContactInfo()
    current_position: CurrentPosition = CurrentPosition()
    experience_years: Optional[int] = None
    top_skills: List[str] = Field(default_factory=list)
    education: Education = Education()
    industry: Optional[str] = None
    career_level: Optional[str] = None
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class ProfileAgentState(MessagesState):
    """
    The agent state for user profile analysis.
    Holds the user_id, prompt, resume text, and conversation messages.
    """
    prompt: Optional[str]
    profile_info: Optional[Dict[str, Any]]


class UserProfileAgent(ReActAgent):
    """Agent for retrieving and analyzing user profile information using ReAct pattern"""

    # List of metadata fields to track
    METADATA_FIELDS = ["profile_info"]

    def __init__(
            self,
            resume_text: str,
            llm: LLM = GroqLLM()
    ):
        """
        Initialize the user profile agent with ReAct capabilities

        Args:
            resume_text: The resume text to analyze
            llm: LLM service for profile analysis
        """
        self.resume_text = resume_text
        super().__init__(llm)

    def _get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        Implements abstract method from ReActAgent.
        """
        return (
            "You are a user profile analysis assistant that helps analyze resume data "
            "and answer questions about user profiles. Follow this process:\n"
            "1. Extract structured profile information using extract_profile_tool\n"
            "2. Answer the user's question or provide a profile summary using profile information\n\n"
        )

    async def invoke(self, prompt: str) -> AgentResponse:
        """
        Process a query about a resume
        Implements abstract method from Agent base class.

        Args:
            prompt: The question or prompt to answer related to the resume

        Returns:
            Dictionary with the response
        """
        try:
            # Initialize state
            initial_state = {
                "prompt": prompt,
                "messages": [
                    {"role": "user", "content": f"Analyze this resume and answer: {prompt}"}
                ],
                "profile_info": None,
            }
            # Initialize metadata fields
            for field in self.METADATA_FIELDS:
                initial_state[field] = None

            result = await self.workflow.ainvoke(initial_state)
            return self._format_response(result)
        except Exception as e:
            return AgentResponse(
                answer=f"An error occurred during profile analysis: {str(e)}",
                error=str(e),
                metadata={"agent_type": "user_profile", "error_type": type(e).__name__}
            )

    def _create_workflow(self) -> CompiledStateGraph:
        """
        Create and return the agent's workflow.
        Implements abstract method from Agent base class.
        """
        builder = StateGraph(ProfileAgentState)
        builder.add_node("think", self.think)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "think")
        builder.add_conditional_edges("think", tools_condition)
        builder.add_edge("tools", "think")
        return builder.compile()

    def _create_tools(self) -> List:
        """
        Creates and returns all the tools for the UserProfileAgent.
        Implements abstract method from ReActAgent.

        Returns:
            List of tools for the ReAct agent.
        """
        return [self._create_extract_profile_tool()]

    def _create_extract_profile_tool(self):
        @tool(parse_docstring=True)
        async def extract_profile_tool(
                keywords: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Extract structured profile information from resume text.

            Args:
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
            {self.resume_text}

            Resume keywords: {", ".join(keywords) if keywords else "No keywords available"}

            Return ONLY JSON conforming to the schema provided.
            """

            response = await self.llm.agenerate(
                system_prompt="You are an expert resume analyst tasked with extracting structured information from resume text.",
                user_message=prompt,
                response_format=ProfileInfoModel,
            )

            if not response.success:
                return {
                    "success": False,
                    "error": f"Failed to extract profile: {response.error}",
                    "profile_info": None,
                }

            return {
                "success": True,
                "profile_info": response.content.model_dump(),
            }

        return extract_profile_tool

    def think(self, state: ProfileAgentState) -> Dict[str, Any]:
        """
        Assistant node that evaluates the current state and executes the next required tool.
        Extends the think method from ReActAgent.
        """
        prompt = state.get("prompt", "")

        # Create a new state dictionary, preserving all existing fields
        new_state = dict(state)

        # Transfer all tracked fields from current state to new state
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_call_id'):
                try:
                    # Try to parse content as JSON if it's a tool response
                    content = json.loads(last_message.content)
                    # Update any matching metadata fields from the content
                    for field in self.METADATA_FIELDS:
                        if field in content:
                            new_state[field] = content[field]
                except (json.JSONDecodeError, AttributeError):
                    pass

        sys_msg = SystemMessage(content=self._get_system_prompt())
        messages: list[BaseMessage] = [sys_msg] + state.get("messages", [])

        # For the first message, add instructions
        if len(state.get("messages", [])) <= 1 and not messages[-1].content.startswith("Here is a candidate's resume"):
            messages.append(
                HumanMessage(
                    content=(
                        f"Here is a candidate's resume:\n{self.resume_text}\n\n"
                        f"Please answer: {prompt}\n\n"
                        "You should extract structured profile information first using extract_profile_tool."
                    )
                )
            )

        invocation = self.llm_with_tools.invoke(messages)
        new_state["messages"] = messages + [invocation]
        return new_state
