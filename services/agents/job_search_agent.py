import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM

load_dotenv()


class JobSearchState(MessagesState):
    """
    State for JobSearchAgent: holds the user prompt, resume data, and conversation messages.
    """
    prompt: str
    resume_text: Optional[str]
    keywords: Optional[List[str]]
    search_results: Optional[List[Dict[str, Any]]]


class JobSearchAgent:
    """
    ReAct agent to perform semantic and filtered searches over job postings in Pinecone.
    """

    def __init__(self, llm: GeminiLLM, pc: Pinecone):
        """
        Initialize JobSearchAgent.

        Args:
            llm: GeminiLLM instance to drive ReAct thinking
            pc: Pinecone client for vector search
        """
        self.llm = llm
        self.pc = pc
        self.index = self.pc.Index("job-postings")

        # Slot to store the true resume embedding for vector search
        self.current_resume_vector: List[float] = []

        # Create all tools for searching and filtering
        self.tools = [
            self._create_vector_similarity_search_tool(),
            # self._create_filter_location_tool(),
            self._create_filter_job_type_tool(),
            self._create_filter_skill_tool(),
            self._create_filter_remote_tool(),
            self._create_filter_date_range_tool(),
            self._create_filter_benefits_tool(),
            self._create_filter_salary_tool(),
            self._create_rank_jobs_tool(),
        ]
        self.llm_with_tools = self.llm.chat.bind_tools(self.tools)

        # Build ReAct agent graph
        builder = StateGraph(JobSearchState)
        builder.add_node("think", self.think)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "think")
        builder.add_conditional_edges("think", tools_condition)
        builder.add_edge("tools", "think")
        self.agent = builder.compile()

    def _create_vector_similarity_search_tool(self):
        @tool(parse_docstring=True)
        def vector_similarity_search_tool(
                top_k: Optional[int] = 20
        ) -> Dict[str, Any]:
            """
            Search for job postings using the resume vector embedding.

            Args:
                top_k: Number of top results to return

            Returns:
                A dict with a "jobs" key containing a list of hits with id, score, and fields.
            """
            rv = self.current_resume_vector
            if not rv:
                raise ValueError("No resume_vector set on agent")
            print("Searching for jobs using vector similarity...", rv)
            response = self.index.query(
                vector=rv,
                top_k=top_k,
                include_metadata=True,
                namespace="jobs",
            )
            hits = response.get("matches", [])
            jobs: List[Dict[str, Any]] = []
            for h in hits:
                entry = {"id": h.get("id"), "score": h.get("score")}
                # Update with metadata instead of fields
                if h.get("metadata"):
                    entry.update(h.get("metadata", {}))
                # Normalize location
                if isinstance(entry.get("location"), dict):
                    loc = entry["location"]
                    entry["locationType"] = loc.get("type", "onsite")
                    entry["locationAddress"] = loc.get("address", "")
                    if loc.get("coordinates"):
                        entry["locationCoordinates"] = loc["coordinates"]
                jobs.append(entry)
            print("Jobs:", jobs)
            return {"jobs": jobs}

        return vector_similarity_search_tool

    def _create_filter_location_tool(self):
        @tool(parse_docstring=True)
        def filter_location_tool(
                jobs: List[Dict[str, Any]],
                location_query: str
        ) -> Dict[str, Any]:
            """
            Filter jobs by location.

            Args:
                jobs: List of job postings
                location_query: Location query string
            """
            print("Filtering jobs by location:", location_query)
            filtered = []
            q = location_query.lower()
            for job in jobs:
                if isinstance(job.get("location"), dict):
                    addr = job["location"].get("address", "").lower()
                    if q in addr:
                        filtered.append(job)
                elif isinstance(job.get("locationAddress"), str):
                    if q in job["locationAddress"].lower():
                        filtered.append(job)
                elif isinstance(job.get("location"), str) and q in job["location"].lower():
                    filtered.append(job)
            return {"jobs": filtered}

        return filter_location_tool

    def _create_filter_job_type_tool(self):
        @tool(parse_docstring=True)
        def filter_job_type_tool(
                jobs: List[Dict[str, Any]],
                jobType: str
        ) -> Dict[str, Any]:
            """
            Filter jobs by job type.

            Args:
                jobs: List of job postings
                jobType: Job type to filter by (e.g., "fulltime", "parttime")
            """
            print("Filtering jobs by job type:", jobType)
            jt = jobType.lower()
            filtered = [j for j in jobs if j.get("jobType", "").lower() == jt]
            return {"jobs": filtered}

        return filter_job_type_tool

    def _create_filter_skill_tool(self):
        @tool(parse_docstring=True)
        def filter_skill_tool(
                jobs: List[Dict[str, Any]],
                skill: str
        ) -> Dict[str, Any]:
            """
            Filter jobs requiring a given skill.

            Args:
                jobs: List of job postings
                skill: Skill to filter by (e.g., "python", "java")
            """
            print("Filtering jobs by skill:", skill)
            sk = skill.lower()
            filtered = []
            for job in jobs:
                skills = job.get("skills", [])
                if isinstance(skills, list) and sk in [s.lower() for s in skills]:
                    filtered.append(job)
                elif sk in job.get("description", "").lower():
                    filtered.append(job)
            return {"jobs": filtered}

        return filter_skill_tool

    def _create_filter_remote_tool(self):
        @tool(parse_docstring=True)
        def filter_remote_tool(
                jobs: List[Dict[str, Any]],
                remote_only: bool = True
        ) -> Dict[str, Any]:
            """
            Filter jobs by remote status.

            Args:
                jobs: List of job postings
                remote_only: If True, return only remote jobs
            """
            print("Filtering jobs by remote status:", remote_only)
            if not remote_only:
                return {"jobs": jobs}
            filtered = []
            for job in jobs:
                loc = job.get("location")
                if isinstance(loc, dict) and loc.get("type", "").lower() == "remote":
                    filtered.append(job)
                elif job.get("locationType", "").lower() == "remote":
                    filtered.append(job)
                elif isinstance(loc, str) and "remote" in loc.lower():
                    filtered.append(job)
            return {"jobs": filtered}

        return filter_remote_tool

    def _create_filter_benefits_tool(self):
        @tool(parse_docstring=True)
        def filter_benefits_tool(
                jobs: List[Dict[str, Any]],
                benefit: str
        ) -> Dict[str, Any]:
            """
            Filter jobs by benefits.

            Args:
                jobs: List of job postings
                benefit: Benefit to filter by (e.g., "health insurance", "401k")
            """
            print("Filtering jobs by benefits:", benefit)
            b = benefit.lower()
            filtered = []
            for job in jobs:
                benefits = job.get("benefits", [])
                if isinstance(benefits, list) and any(b in bi.lower() for bi in benefits):
                    filtered.append(job)
                elif isinstance(benefits, str) and b in benefits.lower():
                    filtered.append(job)
            return {"jobs": filtered}

        return filter_benefits_tool

    def _create_filter_date_range_tool(self):
        @tool(parse_docstring=True)
        def filter_date_range_tool(
                jobs: List[Dict[str, Any]],
                start_date: str,
                end_date: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Filter jobs by posting date range.

            Args:
                jobs: List of job postings
                start_date: Start date in ISO format (YYYY-MM-DD)
                end_date: End date in ISO format (YYYY-MM-DD)
            """
            print("Filtering jobs by date range:", start_date, end_date)
            try:
                sd = datetime.fromisoformat(start_date)
                ed = datetime.fromisoformat(end_date) if end_date else datetime.now()
                filtered = []
                for job in jobs:
                    jd = job.get("date")
                    if not jd:
                        continue
                    if isinstance(jd, str):
                        try:
                            d = datetime.fromisoformat(jd)
                        except ValueError:
                            continue
                    elif isinstance(jd, (int, float)):
                        d = datetime.fromtimestamp(jd)
                    else:
                        continue
                    if sd <= d <= ed:
                        filtered.append(job)
                return {"jobs": filtered}
            except Exception:
                return {"jobs": jobs}

        return filter_date_range_tool

    def _create_filter_salary_tool(self):
        @tool(parse_docstring=True)
        def filter_salary_tool(
                jobs: List[Dict[str, Any]],
                min_salary: Optional[float] = None
        ) -> Dict[str, Any]:
            """
            Filter jobs by minimum salary.

            Args:
                jobs: List of job postings
                min_salary: Minimum salary to filter by
            """
            print("Filtering jobs by minimum salary:", min_salary)
            if min_salary is None:
                return {"jobs": jobs}
            import re
            filtered = []
            for job in jobs:
                text = job.get("salary", "")
                nums = re.findall(r'\d+[,\d]*(?:\.\d+)?', text)
                vals = [float(n.replace(',', '')) for n in nums] if nums else []
                if vals and min(vals) >= min_salary:
                    filtered.append(job)
            return {"jobs": filtered}

        return filter_salary_tool

    def _create_rank_jobs_tool(self):
        @tool(parse_docstring=True)
        def rank_jobs_tool(
                jobs: List[Dict[str, Any]],
                resume_data: Optional[Dict[str, Any]] = None,
                criteria: Optional[Dict[str, float]] = None
        ) -> Dict[str, Any]:
            """
            Rank jobs based on match with resume and criteria.

            Args:
                jobs: List of job postings
                resume_data: Resume data for scoring
                criteria: Scoring criteria (skills, relevance, recency)
            """
            print("Ranking jobs based on criteria:", criteria)
            if not jobs:
                return {"jobs": []}
            crit = criteria or {"skills": 0.5, "relevance": 0.3, "recency": 0.2}
            skills_ref = [k.lower() for k in (resume_data or {}).get("keywords", [])]
            scored = []
            for job in jobs:
                score = job.get("score", 0) * crit["relevance"]
                js = [s.lower() for s in job.get("skills", [])]
                common = set(skills_ref) & set(js)
                if skills_ref:
                    score += (len(common) / len(skills_ref)) * crit["skills"]
                jd = job.get("date")
                if jd:
                    try:
                        pd = datetime.fromisoformat(jd) if isinstance(jd, str) else datetime.fromtimestamp(jd)
                        days_old = (datetime.now() - pd).days
                        score += max(0, 1 - days_old/30) * crit["recency"]
                    except Exception:
                        pass
                job["combined_score"] = score
                scored.append(job)
            ranked = sorted(scored, key=lambda x: x["combined_score"], reverse=True)
            return {"jobs": ranked}

        return rank_jobs_tool

    def think(self, state: JobSearchState) -> Dict[str, Any]:
        """
        Decide whether to call a tool or respond directly.
        """
        sys_msg = SystemMessage(
            content=(
                "You are a job search assistant. Use the available tools to search and refine job listings.\n"
                "1. Optimize the query (if needed)\n"
                "2. Search for jobs using vector similarity\n"
                "3. Apply filters\n"
                "4. Rank jobs"
            )
        )
        messages: List[BaseMessage] = [sys_msg] + state["messages"]
        if len(state["messages"]) == 1:
            messages.append(HumanMessage(content=state["prompt"]))
        invocation = self.llm_with_tools.invoke(messages)
        return {"messages": state["messages"] + [invocation]}

    async def find_jobs(
            self,
            prompt: str,
            resume_data: Optional[Dict[str, Any]] = None,
            top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Run the ReAct graph to search and filter jobs.
        """
        resume_text = resume_data.get("text", "") if resume_data else ""
        keywords = resume_data.get("keywords", []) if resume_data else []
        # Store the real embedding for the vector tool
        self.current_resume_vector = resume_data.get("vector", []) if resume_data else []

        initial_state = {
            "prompt": prompt,
            "resume_text": resume_text,
            "keywords": keywords,
            "messages": [{"role": "user", "content": prompt}]
        }

        result = self.agent.invoke(initial_state)
        final = result["messages"][-1].content
        print("Final result:", final)



if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    llm = GeminiLLM()
    agent = JobSearchAgent(llm=llm, pc=pc)

    async def test_with_resume():
        resume_data = await get_user_resume(
            index=pc.Index("resumes"),
            user_id="wTTRqelPguXTdc5We6v2wRLsjp62"
        )
        resp = await agent.find_jobs(
            "Remote Python developer fulltime",
            resume_data=resume_data,
            top_k=10
        )
        print("Results with resume:", resp)

    asyncio.run(test_with_resume())
