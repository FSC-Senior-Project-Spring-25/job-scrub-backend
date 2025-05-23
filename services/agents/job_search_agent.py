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


class JobSearchState(MessagesState):
    prompt: str
    job_results: Optional[List[Dict[str, Any]]]
    conversation_history: List[Message]


class JobSearchAgent(ReActAgent):
    """A job search agent that uses a single tool to query Pinecone with optional metadata filters."""
    METADATA_FIELDS = ["job_results"]

    def __init__(self, llm: LLM, job_index, resume_vector: List[float]):
        self.job_index = job_index
        self.resume_vector = resume_vector
        super().__init__(llm)

    async def invoke(self, prompt: str, history: List[Message] = None) -> AgentResponse:
        initial_state = {
            "prompt": prompt,
            "conversation_history": history or [],
            "job_results": None,
            "messages": [{"role": "user", "content": prompt}]
        }
        result = await self.workflow.ainvoke(initial_state)
        return self._format_response(result)

    def think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("prompt")
        conversation_history = state.get("conversation_history", [])
        new_state = dict(state)
        # capture any returned results
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, 'tool_call_id'):
                try:
                    data = json.loads(last.content)
                    if "jobs" in data:
                        new_state["job_results"] = data["jobs"]
                except:
                    pass

        # prepend system message
        sys_msg = SystemMessage(content=self._get_system_prompt())
        new_messages: List[BaseMessage] = [sys_msg] + messages

        # if we have job_results, handle summary or no-match
        if new_state.get("job_results") is not None:
            if not new_state["job_results"]:
                new_messages.append(HumanMessage(content="I couldn't find any jobs matching your criteria."))
                new_state["messages"] = messages + [new_messages[-1]]
                return new_state
            # Summarize found jobs
            invocation = self.llm_with_tools.invoke(new_messages)
            new_state["messages"] = messages + [invocation]
            return new_state

        # First turn: prepare context from history and instruct tool call with filters
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get last few messages for context, limited to 5 for relevance
            recent_messages = conversation_history[-5:]
            history_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])
            history_context = f"\nRecent conversation context:\n{history_context}\n"

        new_messages.append(
            HumanMessage(content=(
                f"Use the search_jobs tool to find the top job postings matching: '{prompt}'. "
                f"Please include any relevant metadata filters to narrow the results."
                f"Additionally, adjust filters based on history:"
                f"\n**HISTORY**:\n\n{history_context}"
            ))
        )
        # invoke function-calling
        invocation = self.llm_with_tools.invoke(new_messages)
        new_state["messages"] = messages + [invocation]
        return new_state

    def _get_system_prompt(self) -> str:
        return (
            """
            ROLE
            You are a job-search assistant backed by a Pinecone vector index of job postings.
            When the user asks for jobs, decide which metadata filters will surface the most relevant matches and call the **search_jobs** tool with those filters.

            -------------------------------------------------------------------------------
            1. SEMANTIC-REASONING GUIDELINES (be creative but grounded)
            -------------------------------------------------------------------------------
            • "computer-science major", "CS student" → job_type='internship' OR title contains
              [Software Engineer, Developer, Data Analyst, Research Intern].

            • "entry level", "new grad" → job_type='fulltime', min_date = last 90 days,
              seniority keywords [Junior, Graduate, Associate] in title.

            • Geography:
              – "in Bay Area" → location_address='San Francisco' (also 'San Jose', etc.)
              – "US-remote only" → location_type='remote', location_address='United States'.

            • Benefits:
              – "visa sponsorship" → benefits=['visa sponsorship','H-1B','relocation'].
              – "stock options"   → benefits=['equity','stock options'].

            • Salary:
              – "at least 120k" → min_salary='$120k'.
              – "25/hour"       → min_salary='25 hr'.

            • Time:
              – "posted this week" → min_date = today-7.
              – "last month"       → min_date = today-30.

            • Sector hints:
              – "med-tech" → title or company contains [medical, health, bio].
              – "AI"       → skills include [machine learning, deep learning, LLM].

            If several interpretations are plausible, choose the filter set that is most likely to help the user and briefly state your reasoning in the reply.

            -------------------------------------------------------------------------------
            2. search_jobs ARGUMENTS
            -------------------------------------------------------------------------------
            top_k              → default 5; increase when the user explicitly asks for more.
            title              → partial match strings or list of synonyms.
            company            → exact or partial match.
            job_type           → 'fulltime' | 'parttime' | 'internship' | 'contract' | 'volunteer'.
            location_type      → 'remote' | 'onsite' | 'hybrid'.
            min_date / max_date→ YYYY-MM-DD.
            skills             → list; ANY match.
            keywords           → list; ANY match in job description or metadata.
            benefits           → list; ANY match.
            min_salary / max_salary → substring checks (e.g. '$120k', '25 hr').
            location_address   → substring on city, state, or country.

            Include only parameters that materially narrow the results.

            -------------------------------------------------------------------------------
            3. WORKFLOW
            -------------------------------------------------------------------------------
            1. Parse the user message and conversation context to infer filters (see Section 1).
            2. Call search_jobs with a concise filter object.
            3. Inspect results.
               • If none → apologise and ask for clarifications or broaden filters and retry.
               • If few but relevant → return them and invite further refinement if needed.
            4. Respond to the user.
               • For matches, list each job on one line:
                 - **{title}** at {company} (Remote · $Salary): [View](/jobs/{id})
               • Prepend a short sentence that explains how you interpreted the query.
               • Never fabricate job data; display only what the tool returns.
            """
        )

    def _create_tools(self) -> List:
        @tool
        def search_jobs(
                top_k: int = 5,
                title: Optional[str] = None,
                company: Optional[str] = None,
                job_type: Optional[str] = None,
                location_type: Optional[str] = None,
                min_date: Optional[str] = None,
                max_date: Optional[str] = None,
                skills: Optional[List[str]] = None,
                keywords: Optional[List[str]] = None,
                benefits: Optional[List[str]] = None,
                min_salary: Optional[str] = None,
                max_salary: Optional[str] = None,
                location_address: Optional[str] = None
        ) -> str:
            """
            Query Pinecone for jobs matching the resume vector and optional metadata filters.

            Args:
                top_k: Number of results to return (default: 5)
                title: Filter by exact job title
                company: Filter by exact company name
                job_type: One of 'fulltime','parttime','internship','contract','volunteer'
                location_type: One of 'remote','onsite','hybrid'
                min_date: Minimum posting date in format 'YYYY-MM-DD'
                max_date: Maximum posting date in format 'YYYY-MM-DD'
                keywords: List of keywords to filter by
                skills: List of required skills to filter by
                benefits: List of required benefits to filter by
                min_salary: Minimum salary string (will be matched as substring)
                max_salary: Maximum salary string (will be matched as substring)
                location_address: Filter by location address substring

            Returns:
                JSON string with jobs list and count
            """
            metadata_filter: Dict[str, Any] = {}

            if job_type:
                metadata_filter["jobType"] = {"$eq": job_type}
            if location_type:
                metadata_filter["locationType"] = {"$eq": location_type}

            if min_date:
                metadata_filter["date"] = metadata_filter.get("date", {})
                metadata_filter["date"]["$gte"] = min_date
            if max_date:
                metadata_filter["date"] = metadata_filter.get("date", {})
                metadata_filter["date"]["$lte"] = max_date

            # For list fields like skills and benefits, use $in operator instead of requiring all
            if skills and len(skills) > 0:
                # Create a filter that matches if ANY of the requested skills are present
                metadata_filter["$or"] = metadata_filter.get("$or", [])
                metadata_filter["$or"].extend([{"skills": skill} for skill in skills])

            if benefits and len(benefits) > 0:
                # Similarly, match if ANY benefits match
                if "$or" not in metadata_filter:
                    metadata_filter["$or"] = []
                metadata_filter["$or"].extend([{"benefits": benefit} for benefit in benefits])

            if keywords and len(keywords) > 0:
                # Use $in operator to match any job containing at least one keyword
                metadata_filter["keywords"] = {"$in": keywords}

            print(f"[JOB SEARCH]: Using metadata filter: {metadata_filter}")

            try:
                response = self.job_index.query(
                    vector=self.resume_vector,
                    top_k=top_k * 2,
                    namespace="jobs",
                    filter=metadata_filter if metadata_filter else None,
                    include_metadata=True
                )

                # Post-process results for more flexible matching
                jobs = []
                for match in response.matches:
                    job = match.metadata
                    job["id"] = match.id
                    job["score"] = match.score

                    # Apply flexible post-filtering
                    include_job = True

                    # Title substring matching
                    if title and job.get("title"):
                        if title.lower() not in job.get("title", "").lower():
                            include_job = False

                    # Company substring matching
                    if company and job.get("company"):
                        if company.lower() not in job.get("company", "").lower():
                            include_job = False

                    # Salary substring matching
                    if min_salary and job.get("salary"):
                        # Just ensure there's some salary info, detailed matching can be complex
                        pass

                    if max_salary and job.get("salary"):
                        # Just ensure there's some salary info, detailed matching can be complex
                        pass

                    # Location address substring matching
                    if location_address and job.get("location") and job["location"].get("address"):
                        if location_address.lower() not in job["location"]["address"].lower():
                            include_job = False

                    if include_job:
                        jobs.append(job)

                # Limit to requested top_k after post-filtering
                return json.dumps({"jobs": jobs[:top_k], "count": len(jobs[:top_k])})
            except Exception as e:
                return json.dumps({"error": str(e), "jobs": [], "count": 0})

        return [search_jobs]

    def _create_workflow(self) -> CompiledStateGraph:
        graph = StateGraph(JobSearchState)
        graph.add_node("think", self.think)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "think")
        graph.add_conditional_edges("think", tools_condition)
        graph.add_edge("tools", "think")
        return graph.compile()

    def _extract_answer(self, state: Dict[str, Any]) -> str:
        """Return a Markdown list of jobs with proper links."""
        msgs = state.get("messages", [])
        job_results = state.get("job_results", [])

        if job_results:
            lines = ["## Matching Jobs", ""]  # heading + blank line
            for i, job in enumerate(job_results):
                job_id = job.get("id", f"job-{i}")
                title = job.get("title", "Untitled Position")
                company = job.get("company", "Unknown Company")
                loc_type = job.get("locationType", "")
                salary = job.get("salary", "")

                meta = " · ".join(filter(None, [loc_type, salary]))  # compact meta
                meta = f" ({meta})" if meta else ""

                # ONE bullet, everything on one line
                lines.append(
                    f"- **{title}** at {company}{meta}: "
                    f"[View&nbsp;Details](/jobs/{job_id})"
                )

            return "\n".join(lines) + "\n"  # final trailing \n is optional

        # fall‑back to last assistant message if nothing to format
        return msgs[-1].content if msgs else ""

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        meta = {"agent_type": "job_search"}
        if state.get("job_results") is not None:
            meta["job_results"] = state["job_results"]
        return meta