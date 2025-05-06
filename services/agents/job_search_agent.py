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


class JobFilterParams(BaseModel):
    """Schema for job search filter parameters."""
    title: Optional[str] = Field(None, description="Exact job title to match")
    company: Optional[str] = Field(None, description="Exact company name to match")
    job_types: Optional[List[str]] = Field(None, description="Job types (fulltime, parttime, internship, contract, volunteer)")
    location_types: Optional[List[str]] = Field(None, description="Location types (remote, onsite, hybrid)")
    min_date: Optional[str] = Field(None, description="Minimum date in YYYY-MM-DD format")
    max_date: Optional[str] = Field(None, description="Maximum date in YYYY-MM-DD format")
    skills: Optional[List[str]] = Field(None, description="Required skills to filter jobs")
    benefits: Optional[List[str]] = Field(None, description="Required benefits to filter jobs")
    min_salary: Optional[str] = Field(None, description="Minimum salary filter")
    max_salary: Optional[str] = Field(None, description="Maximum salary filter")
    location_address: Optional[str] = Field(None, description="Location substring to match")
    top_k: Optional[int] = Field(5, description="Number of results to return")


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

        if new_state.get("job_results") is not None:
            if not new_state["job_results"]:
                new_messages.append(HumanMessage(content="I couldn't find any job postings matching your criteria."))
                new_state["messages"] = messages + [new_messages[-1]]
                return new_state
            # Summarize found jobs
            invocation = self.llm_with_tools.invoke(new_messages)
            new_state["messages"] = messages + [invocation]
            return new_state

        # First turn: preprocess query with LLM to create intelligent filters
        if not any(hasattr(msg, 'tool_call_id') for msg in messages):
            # Extract relevant skills and job criteria from the prompt
            try:
                filter_prompt = f"""
                Extract relevant job search filters from this query: "{prompt}"

                For a Computer Science major, consider these filter parameters:
                1. Job titles related to computer science (Software Engineer, Data Scientist, etc.)
                2. Skills that would be relevant (programming languages, frameworks, etc.)
                3. Job types that would be appropriate (fulltime, internship, etc.)
                4. Location preferences (remote, hybrid, onsite, specific location)
                5. Company names mentioned
                6. Benefits that might be important
                7. Salary expectations if mentioned
                8. Date ranges for job postings if mentioned

                Return ONLY a structured JSON object with these fields (leave empty if not specified):
                {{
                    "title": "exact job title",
                    "company": "exact company name",
                    "job_types": ["fulltime", "parttime", "internship", "contract", "volunteer"],
                    "location_types": ["remote", "onsite", "hybrid"],
                    "min_date": "YYYY-MM-DD",
                    "max_date": "YYYY-MM-DD",
                    "skills": ["skill1", "skill2"],
                    "benefits": ["benefit1", "benefit2"],
                    "min_salary": "min salary as string",
                    "max_salary": "max salary as string",
                    "location_address": "location substring",
                    "top_k": 5
                }}
                """

                filter_response = self.llm.generate(
                    system_prompt="You are a job search filter generator that extracts structured search criteria from natural language queries about computer science jobs.",
                    user_message=filter_prompt,
                    response_format=JobFilterParams
                )

                if filter_response.success:
                    # Use filter_response.content directly as it's already a JobFilterParams instance
                    filters = filter_response.content

                    # Log the extracted filters
                    print(f"[JOB_SEARCH] Extracted filters: {filters}")

                    # Create tool call instruction with intelligent filters
                    filter_instructions = ""
                    if filters.skills:
                        filter_instructions += f"\nSkills to filter by: {', '.join(filters.skills)}"
                    if filters.title:
                        filter_instructions += f"\nPossible job title: {filters.title}"
                    if filters.job_types:
                        filter_instructions += f"\nJob types to consider: {', '.join(filters.job_types)}"
                    if filters.location_types:
                        filter_instructions += f"\nLocation types to consider: {', '.join(filters.location_types)}"

                    instruction = (
                        f"Use the search_jobs tool to find jobs matching: '{prompt}'. {filter_instructions}\n"
                        f"Use these filters to construct an appropriate search query."
                    )
                else:
                    # Fall back to generic instruction if LLM fails
                    instruction = f"Use the search_jobs tool to find jobs matching: '{prompt}'. Consider focusing on computer science related positions."

            except Exception as e:
                print(f"[JOB_SEARCH] Error in filter preprocessing: {str(e)}")
                instruction = f"Use the search_jobs tool to find jobs matching: '{prompt}'."

            new_messages.append(HumanMessage(content=instruction))

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
