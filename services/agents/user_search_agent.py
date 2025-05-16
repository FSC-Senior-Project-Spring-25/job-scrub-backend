import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from models.chat import Message
from services.agents.base.agent import ReActAgent, AgentResponse
from services.llm.base.llm import LLM


class UserSearchState(MessagesState):
    prompt: str
    conversation_history: List[Message]
    user_results: Optional[List[Dict[str, Any]]]


class UserSearchAgent(ReActAgent):
    """Agent that searches for similar users by resume content in Pinecone."""
    METADATA_FIELDS = ["user_results"]

    def __init__(self, llm: LLM, resume_index, resume_vector: List[float]):
        self.resume_index = resume_index
        self.resume_vector = resume_vector
        super().__init__(llm)

    async def invoke(self, prompt: str, history: List[Message]) -> AgentResponse:
        initial_state = {
            "prompt": prompt,
            "conversation_history": history or [],
            "user_results": None,
            "messages": [{"role": "user", "content": prompt}]
        }
        result = await self.workflow.ainvoke(initial_state)
        return self._format_response(result)

    def think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("prompt")
        conversation_history = state.get("conversation_history", [])  # Retrieve history
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

        # First turn: include recent conversation context and instruct tool call
        history_context = ""
        if conversation_history:
            recent = conversation_history[-5:]
            history_context = "\nRecent conversation context:\n" + "\n".join(f"{m.role}: {m.content}" for m in recent) + "\n"

        new_messages.append(
            HumanMessage(content=(
                f"Use the search_users tool to find users with resumes matching: '{prompt}'. "
                f"Please include any relevant metadata filters to narrow the results."
                f"Additionally, adjust filters based on history:"
                f"\n**HISTORY**:\n\n{history_context}"
            ))
        )
        # Invoke function-calling
        invocation = self.llm_with_tools.invoke(new_messages)
        new_state["messages"] = messages + [invocation]
        return new_state

    def _get_system_prompt(self) -> str:
        return (
            """
            ROLE  
            You are a user-search assistant backed by a Pinecone index of résumé vectors.  
            Your goal is to translate the user’s natural-language query into an optimal **search_users** tool call and present concise results.
            
            ---------------------------------------------------------------------------
            1. SEMANTIC-REASONING GUIDELINES
            ---------------------------------------------------------------------------
            • Degree or major cues:  
              “computer-science majors”, “CS students” → keywords=['computer science', 'cs', 'software'].
            
            • Skill cues:  
              “python developers” → keywords=['python'] (plus related stack e.g. 'django', 'pandas').  
              “machine-learning experience” → keywords=['machine learning', 'tensorflow', 'pytorch'].
            
            • Seniority cues:  
              “entry-level” → keywords=['junior', 'intern', 'graduate'].  
              “senior” → keywords=['senior', 'lead', 'principal'].
            
            • Text snippets:  
              If the user supplies an exact phrase (“experience with FDA regulation”), pass it via text_contains.
            
            • File hints:  
              “latest résumé” or filename mention (“resume-2025.pdf”) → filename filter.
            
            Infer reasonable synonyms and related terms; prefer inclusive keyword lists so any match counts.
            
            ---------------------------------------------------------------------------
            2. search_users ARGUMENTS
            ---------------------------------------------------------------------------
            top_k               → default 5; increase if the user explicitly asks for more.  
            keywords            → list of terms; ANY match in résumé metadata.  
            text_contains       → substring that must appear in résumé full text.  
            filename            → exact résumé filename (use only if user specifies it).
            
            Supply only parameters that materially narrow results.
            
            ---------------------------------------------------------------------------
            3. WORKFLOW
            ---------------------------------------------------------------------------
            1. Parse the query, infer filters (see Section 1).  
            2. Call search_users with a concise filter object.  
            3. Evaluate results.  
               • No matches → apologise and ask for refinements or broaden filters and retry.  
               • Matches → proceed.  
            4. Respond.  
               • Begin with one sentence explaining how the query was interpreted.  
               • List each user on one line:  
                 - **{name}** : {short summary} – [View Profile](/profile/{id})  
               • If available, append a “Resume Summaries” section with up to 200 characters from each résumé.  
               • Never fabricate user data; show only what the tool returns.
               • Remove any duplicate users.
            """
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

            users = []
            print(f"[USER SEARCH]: Using metadata filter: {metadata_filter}")
            for match in response.matches:
                user_data = {
                    "user_id": match.id,
                    "score": match.score,
                    "keywords": match.metadata.get("keywords", []),
                    "file_id": match.metadata.get("file_id", ""),
                    "text": match.metadata.get("text", ""),
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
        user_results = state.get("user_results", [])
        if user_results:
            lines = ["## Found Users", ""]  # heading + blank line
            for i, user in enumerate(user_results):
                user_id = user.get("user_id", f"user-{i}")
                keywords = user.get("keywords", [])
                text_summary = user.get("text", "")

                # Create a brief summary from keywords and text
                keyword_text = ", ".join(keywords[:3]) if keywords else ""
                summary = f"{keyword_text}"
                if text_summary:
                    summary += f": {text_summary[:100]}..." if len(text_summary) > 100 else f": {text_summary}"

                # ONE bullet, everything on one line
                lines.append(
                    f"- {summary} – [View Profile](/profile/{user_id})"
                )

            # Append last message content
            if msgs:
                lines.append("")  # blank line
                lines.append(msgs[-1].content)

            return "\n".join(lines) + "\n"  # final trailing \n is optional

        # fall‑back to last assistant message if nothing to format
        return msgs[-1].content if msgs else ""

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        meta = {"agent_type": "user_search"}
        if state.get("user_results") is not None:
            meta["user_results"] = state["user_results"]
        return meta
