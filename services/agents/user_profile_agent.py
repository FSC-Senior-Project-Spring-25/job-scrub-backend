import json
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from services.agents.base.agent import ReActAgent, AgentResponse
from services.gemini import GeminiLLM, ResponseFormat

load_dotenv()


class ProfileAgentState(MessagesState):
    """
    The agent state for user profile analysis.
    Holds the user_id, prompt, resume text, and conversation messages.
    """
    resume_text: Optional[str]
    prompt: Optional[str]
    profile_info: Optional[Dict[str, Any]]


class UserProfileAgent(ReActAgent):
    """Agent for retrieving and analyzing user profile information using ReAct pattern"""

    # List of metadata fields to track
    METADATA_FIELDS = ["profile_info"]

    def __init__(self, llm: GeminiLLM = GeminiLLM()):
        """
        Initialize the user profile agent with ReAct capabilities

        Args:
            llm: LLM service for profile analysis
        """
        super().__init__(llm)

        print("==" * 20)
        print("User Profile Agent Graph:")
        print("==" * 20)
        self.workflow.get_graph().print_ascii()

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

    async def invoke(self, **kwargs) -> AgentResponse:
        """
        Process a query about a resume
        Implements abstract method from Agent base class.

        Args:
            resume_text: The resume text to analyze
            prompt: The question or prompt to answer related to the resume

        Returns:
            Dictionary with the response
        """
        resume_text = kwargs.get("resume_text", "")
        prompt = kwargs.get("prompt", "")

        try:
            # Initialize state
            initial_state = {
                "resume_text": resume_text,
                "prompt": prompt,
                "messages": [
                    {"role": "user", "content": f"Analyze this resume and answer: {prompt}"}
                ],
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

    def _extract_answer(self, state: Dict[str, Any]) -> str:
        """
        Extract final answer from result.
        Overrides method from Agent base class.
        """
        # Use the parent implementation to extract the answer
        answer = super()._extract_answer(state)
        print("Final User Profile Message:", answer)
        return answer

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from the state.
        Overrides method from Agent base class.

        Returns:
            Dictionary with metadata fields
        """
        metadata = {"agent_type": "user_profile"}
        print("Extract Metadata Fields:", state)
        # Add all tracked fields to metadata
        for field in self.METADATA_FIELDS:
            if field in state:
                metadata[field] = state.get(field)

        print("Extracted Metadata:", metadata)
        return metadata

    def _create_workflow(self) -> CompiledStateGraph:
        """
        Create and return the agent's workflow.
        Implements abstract method from Agent base class.
        """
        # Build the ReAct agent state graph
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
        Extends the think method from ReActAgent.
        """
        resume_text = state.get("resume_text", "")
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
                        f"Here is a candidate's resume:\n{resume_text}\n\n"
                        f"Please answer: {prompt}\n\n"
                        "You should extract structured profile information first using extract_profile_tool."
                    )
                )
            )

        invocation = self.llm_with_tools.invoke(messages)
        new_state["messages"] = messages + [invocation]
        return new_state
