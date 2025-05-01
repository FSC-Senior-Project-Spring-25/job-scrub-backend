import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from services.agents.base.agent import ReActAgent, AgentResponse
from services.gemini import GeminiLLM


class UserSearchState(MessagesState):
    prompt: str
    user_results: Optional[List[Dict[str, Any]]]


class UserSearchAgent(ReActAgent):
    """Agent that searches for similar users by resume content in Pinecone."""
    METADATA_FIELDS = ["user_results"]

    def __init__(self, llm: GeminiLLM, resume_index, resume_vector: List[float]):
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
                text_contains: Optional[str] = None,
        ) -> str:
            """
            Search for users with resumes similar to the query vector with optional metadata filtering.

            Args:
                top_k: Number of results to return (default: 5)
                keywords: List of keywords; matches any resume containing at least one keyword
                text_contains: Text that must appear somewhere in the resume (post-filtered)

            Returns:
                JSON string with users list and count
            """
            # Build a more tolerant metadata filter
            metadata_filter: Dict[str, Any] = {}
            if keywords:
                # match any resume whose `keywords` metadata array contains at least one of the requested keywords
                metadata_filter["keywords"] = {"$in": keywords}

            # First attempt: vector + metadata filter
            def _run_query(filter_expr):
                return self.resume_index.query(
                    vector=self.resume_vector,
                    top_k=top_k,
                    namespace="resumes",
                    filter=filter_expr,
                    include_metadata=True
                )

            # try with filter, but if no matches come back, drop the filter entirely
            response = _run_query(metadata_filter if metadata_filter else None)
            if not response.matches and metadata_filter:
                # fallback to pure vector search
                response = _run_query(None)

            print(f"Search results: {response.matches}")
            users = []
            for match in response.matches:
                user_data = {
                    "user_id": match.id,
                    "score": match.score,
                    "keywords": match.metadata.get("keywords", []),
                    "file_id": match.metadata.get("file_id", ""),
                }

                # summarize the text field if present
                full_text = match.metadata.get("text", "")
                user_data["text_summary"] = (
                    full_text[:200] + "..." if len(full_text) > 200 else full_text
                )

                # post-filter for text_contains
                if text_contains:
                    if text_contains.lower() not in full_text.lower():
                        continue

                users.append(user_data)

            return json.dumps({"users": users, "count": len(users)})

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
        msgs = state.get("messages", [])
        base_answer = msgs[-1].content if msgs else ""

        # Check if we have user results with text summaries to include
        user_results = state.get("user_results")
        if user_results:
            # Extract text summaries
            summaries = []
            for idx, user in enumerate(user_results[:5], 1):  # Limit to first 5 users
                if "text_summary" in user:
                    summary = user["text_summary"]
                    # Truncate if too long
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    summaries.append(f"**User {idx}**: {summary}")

            # Add summaries to the answer if available
            if summaries:
                summary_section = "\n\n**Resume Summaries**:\n" + "\n\n".join(summaries)
                return base_answer + summary_section

        return base_answer

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        meta = {"agent_type": "user_search"}
        if state.get("user_results") is not None:
            meta["user_results"] = state["user_results"]
        return meta
