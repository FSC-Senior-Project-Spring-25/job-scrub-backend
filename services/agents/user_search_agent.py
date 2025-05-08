import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from services.agents.base.agent import ReActAgent, AgentResponse
from services.llm.base.llm import LLM


class ResumeProfile(BaseModel):
    """Schema for structured resume profile information."""
    name: str = Field(default="Candidate", description="Person's full name")
    title: str = Field(default="Professional", description="Current or most recent job title")
    skills: List[str] = Field(default_factory=list, description="Key professional skills")
    education: Optional[str] = Field(None, description="Highest education level and field")
    experience_years: Optional[int] = Field(None, description="Total years of professional experience")
    industry: Optional[str] = Field(None, description="Primary industry of expertise")
    contact_info: Optional[str] = Field(None, description="Email, phone, or other contact information")


class UserSearchState(MessagesState):
    prompt: str
    user_results: Optional[List[Dict[str, Any]]]


class UserSearchAgent(ReActAgent):
    """Agent that searches for similar users by resume content in Pinecone."""
    METADATA_FIELDS = ["user_results"]

    def __init__(self, llm: LLM, resume_index, resume_vector: List[float]):
        self.resume_index = resume_index
        self.resume_vector = resume_vector
        super().__init__(llm)

    async def invoke(self, prompt: str) -> AgentResponse:
        initial_state = {
            "prompt": prompt,
            "user_results": None,
            "messages": [{"role": "user", "content": prompt}]
        }
        result = await self.workflow.ainvoke(initial_state)
        return self._format_response(result)

    def think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("prompt")
        new_state = dict(state)

        # Capture any returned results
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, 'tool_call_id'):
                try:
                    data = json.loads(last.content)
                    if "users" in data:
                        new_state["user_results"] = data["users"]
                except:
                    pass

        # Prepend system message
        sys_msg = SystemMessage(content=self._get_system_prompt())
        new_messages: List[BaseMessage] = [sys_msg] + messages

        # If we have user_results, handle summary or no-match
        if new_state.get("user_results") is not None:
            if not new_state["user_results"]:
                new_messages.append(HumanMessage(content="I couldn't find any users with resumes matching your criteria."))
                new_state["messages"] = messages + [new_messages[-1]]
                return new_state
            # Summarize found users
            invocation = self.llm_with_tools.invoke(new_messages)
            new_state["messages"] = messages + [invocation]
            return new_state

        # First turn: instruct tool call with filters
        new_messages.append(
            HumanMessage(content=(
                f"Use the search_users tool to find users with resumes matching: '{prompt}'. "
                "You can use filters based on resume keywords and content."
            ))
        )
        # Invoke function-calling
        invocation = self.llm_with_tools.invoke(new_messages)
        new_state["messages"] = messages + [invocation]
        return new_state

    def _get_system_prompt(self) -> str:
        return (
            "You are a user search assistant. You can find users with resumes that match specific criteria using "
            "the search_users tool. The tool accepts these metadata filters:\n"
            "- keywords: Filter by specific keywords in the user's resume\n"
            "- text_contains: Filter by text content in the user's resume\n"
            "- filename: Filter by exact resume filename\n"
            "- top_k: Number of results to return (default: 5)\n\n"
            "Users are found by semantic similarity to the provided resume vector. "
            "The filters help narrow down results. Don't invent user profiles or data."
        )

    def _create_tools(self) -> List:
        @tool
        def search_users(
                top_k: int = 5,
                keywords: Optional[List[str]] = None,
                text_contains: Optional[str] = None
        ) -> str:
            """
            Search for users with resumes similar to the query vector with optional metadata filtering.

            Args:
                top_k: Number of results to return (default: 5)
                keywords: List of keywords to search for in the user's resume
                text_contains: Text that must appear somewhere in the resume (post-filtered)

            Returns:
                JSON string with users list and count
            """
            # Build metadata filter for keywords
            metadata_filter: Dict[str, Any] = {}

            # For keywords, use $in operator to match any resume containing at least one keyword
            if keywords and len(keywords) > 0:
                metadata_filter["keywords"] = {"$in": keywords}

            print(f"[USER SEARCH]: Using metadata filter: {metadata_filter}")

            try:
                # Query with vector similarity, fetch more to allow for post-filtering
                response = self.resume_index.query(
                    vector=self.resume_vector,
                    top_k=top_k * 2,  # Fetch more to allow for post-filtering
                    namespace="resumes",
                    filter=metadata_filter if metadata_filter else None,
                    include_metadata=True
                )

                # Post-process results
                users = []
                for match in response.matches:
                    # Extract basic user data
                    user_data = {
                        "user_id": match.id,
                        "score": match.score,
                        "keywords": match.metadata.get("keywords", []),
                        "file_id": match.metadata.get("file_id", ""),
                        "text": match.metadata.get("text", ""),
                    }

                    # Post-filtering for text content
                    include_user = True
                    full_text = match.metadata.get("text", "")

                    # Text contains filtering - case insensitive
                    if text_contains and text_contains.lower() not in full_text.lower():
                        include_user = False

                    if include_user:
                        users.append(user_data)

                print(f"[USER SEARCH]: {len(users)} users found after filtering")
                return json.dumps({"users": users[:top_k], "count": len(users[:top_k])})

            except Exception as e:
                print(f"[USER SEARCH ERROR]: {str(e)}")
                return json.dumps({"error": str(e), "users": [], "count": 0})

        return [search_users]

    def _create_workflow(self) -> CompiledStateGraph:
        graph = StateGraph(UserSearchState)
        graph.add_node("think", self.think)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "think")
        graph.add_conditional_edges("think", tools_condition)
        graph.add_edge("tools", "think")
        return graph.compile()

    def _extract_answer(self, state: Dict[str, Any]) -> str:
        """
        Return a Markdown list of users with structured profile information and links.
        Uses LLM to extract profile details from resume text when available.
        """
        msgs = state.get("messages", [])
        user_results = state.get("user_results", [])

        if not user_results:
            return msgs[-1].content if msgs else ""

        lines = ["## Matching Candidates", ""]

        for i, user in enumerate(user_results):
            user_id = user.get("user_id", f"user-{i}")
            keywords = user.get("keywords", [])

            # Important: Get the full resume text - this is what was missing before
            full_text = user.get("text", "")

            # Default values in case LLM fails
            name = "User"
            title = None
            skills = keywords[:5] if keywords else []
            contact = "Contact via profile"
            education = None
            experience = None

            # Only invoke LLM if we have text to analyze
            if full_text:
                try:
                    prompt = f"""
                    Extract key professional information from this resume excerpt:

                    {full_text[:1500]}

                    Keywords found in metadata: {', '.join(keywords) if keywords else 'None'}
                    """

                    # Actually call the LLM
                    response = self.llm.generate(
                        system_prompt="You are an expert resume analyst. Extract structured information including contact details.",
                        user_message=prompt,
                        response_format=ResumeProfile
                    )

                    if response.success:
                        profile = response.content
                        name = profile.name if profile.name != "Candidate" else "Professional Candidate"
                        title = profile.title if profile.title != "Professional" else "Skilled Professional"
                        skills = profile.skills if profile.skills else keywords[:5]
                        contact = profile.contact_info if profile.contact_info else "Contact via profile"
                        education = profile.education
                        experience = f"{profile.experience_years} years" if profile.experience_years else None
                except Exception as e:
                    print(f"[USER_SEARCH] Error extracting profile from LLM: {str(e)}")

            # Format skills as a comma-separated list
            skills_str = ", ".join(skills[:5])
            if len(skills) > 5:
                skills_str += "..."

            # Build a more informative and structured candidate entry
            entry = [f"### {name}"]

            # Title on one line
            if title:
                entry.append(f"**{title}** ")

            # Skills section
            entry.append(f"**Skills**: {skills_str}")

            # Optional sections when available
            if education:
                entry.append(f"**Education**: {education}")

            if experience:
                entry.append(f"**Experience**: {experience}")

            if contact and contact != "Contact via profile":
                entry.append(f"**Contact**: {contact}")

            # Add profile link
            entry.append(f"[View Full Profile](/profile/{user_id})")

            # Add all lines with proper spacing
            lines.extend(entry)
            lines.append("")  # Add spacing between candidates

        return "\n".join(lines)

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        meta = {"agent_type": "user_search"}
        if state.get("user_results") is not None:
            meta["user_results"] = state["user_results"]
        return meta
