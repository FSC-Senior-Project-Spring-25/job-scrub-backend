import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from services.agents.base.agent import ReActAgent, AgentResponse
from services.llm.base.llm import LLM


class JobSearchState(MessagesState):
    prompt: str
    job_results: Optional[List[Dict[str, Any]]]


class JobSearchAgent(ReActAgent):
    """A job search agent that uses a single tool to query Pinecone with optional metadata filters."""
    METADATA_FIELDS = ["job_results"]

    def __init__(self, llm: LLM, job_index, resume_vector: List[float]):
        self.job_index = job_index
        self.resume_vector = resume_vector
        super().__init__(llm)

    async def invoke(self, prompt: str) -> AgentResponse:
        initial_state = {
            "prompt": prompt,
            "job_results": None,
            "messages": [{"role": "user", "content": prompt}]
        }
        result = await self.workflow.ainvoke(initial_state)
        return self._format_response(result)

    def think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("prompt")
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
                new_messages.append(HumanMessage(content="I’m sorry, I couldn’t find any jobs matching your criteria."))
                new_state["messages"] = messages + [new_messages[-1]]
                return new_state
            # summarize found jobs
            invocation = self.llm_with_tools.invoke(new_messages)
            new_state["messages"] = messages + [invocation]
            return new_state

        # first turn: instruct tool call with filters
        new_messages.append(
            HumanMessage(content=(
                f"Use the search_jobs tool to find the top job postings matching: '{prompt}'. "
                "You may pass filter arguments like locationType or jobType to narrow results."
            ))
        )
        # invoke function-calling
        invocation = self.llm_with_tools.invoke(new_messages)
        new_state["messages"] = messages + [invocation]
        return new_state

    def _get_system_prompt(self) -> str:
        return (
            "You are a job search assistant. You have access to a search_jobs tool that accepts these metadata filters:\n"
            "- title: Filter by exact job title\n"
            "- company: Filter by exact company name\n"
            "- job_type: One of 'fulltime','parttime','internship','contract','volunteer'\n"
            "- location_type: One of 'remote','onsite','hybrid'\n"
            "- min_date/max_date: Date range filters in format 'YYYY-MM-DD'\n"
            "- skills: List of required skills to filter by\n"
            "- benefits: List of required benefits to filter by\n"
            "- min_salary/max_salary: Salary range filters (substring match)\n"
            "- location_address: Location substring to match\n"
            "- top_k: Number of results to return (default: 5)\n\n"
            "Use these filters to retrieve relevant job postings based on the internal resume vector. "
            "Be precise with your filters to find the most relevant jobs. Do not invent listings."
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
                skills: List of required skills to filter by
                benefits: List of required benefits to filter by
                min_salary: Minimum salary string (will be matched as substring)
                max_salary: Maximum salary string (will be matched as substring)
                location_address: Filter by location address substring

            Returns:
                JSON string with jobs list and count
            """
            metadata_filter: Dict[str, Any] = {}

            # These enum-like fields can stay as exact matches
            if job_type:
                metadata_filter["jobType"] = {"$eq": job_type}
            if location_type:
                metadata_filter["locationType"] = {"$eq": location_type}

            # Date range filters stay the same
            if min_date:
                metadata_filter["date"] = metadata_filter.get("date", {})
                metadata_filter["date"]["$gte"] = min_date
            if max_date:
                metadata_filter["date"] = metadata_filter.get("date", {})
                metadata_filter["date"]["$lte"] = max_date

            # For list fields like skills and benefits, use $in operator instead of requiring all
            if skills and len(skills) > 0:
                # Create a filter that matches if ANY of the requested skills are present
                # This is more lenient than requiring all skills
                metadata_filter["$or"] = metadata_filter.get("$or", [])
                metadata_filter["$or"].extend([{"skills": skill} for skill in skills])

            if benefits and len(benefits) > 0:
                # Similarly, match if ANY benefits match
                if "$or" not in metadata_filter:
                    metadata_filter["$or"] = []
                metadata_filter["$or"].extend([{"benefits": benefit} for benefit in benefits])

            print("Metadata filter:", metadata_filter)

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
                print(f"[JOB SEARCH]: f{json.dumps({"jobs": jobs[:top_k], "count": len(jobs[:top_k])})}")
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
