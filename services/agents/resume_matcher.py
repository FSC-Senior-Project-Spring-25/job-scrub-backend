from typing import TypedDict
import torch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from services.agents.tools.extract_keywords import extract_keywords
from services.gemini import GeminiLLM
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder


class MatchingState(TypedDict):
    resume_text: str
    job_text: str
    resume_keywords: list[str]
    job_keywords: list[str]
    missing_keywords: list[str]
    match_score: float
    keyword_coverage: float
    similarity_details: list[dict]
    resume_bytes: bytes


class ResumeMatchingAgent:
    def __init__(
            self,
            resume_parser: ResumeParser,
            text_embedder: TextEmbedder,
            llm: GeminiLLM
    ):
        self.resume_parser = resume_parser
        self.text_embedder = text_embedder
        self.llm = llm
        self.workflow = self._create_workflow()

    async def analyze_resume(self, resume_bytes: bytes, job_description: str) -> dict:
        """Analyze resume from PDF bytes"""
        initial_state = MatchingState(
            resume_text="",
            job_text=job_description,
            resume_keywords=[],
            job_keywords=[],
            missing_keywords=[],
            match_score=0.0,
            keyword_coverage=0.0,
            similarity_details=[],
            resume_bytes=resume_bytes
        )

        result = await self.workflow.ainvoke(initial_state)
        return self._format_result(result)

    async def analyze_resume_text(self, resume_text: str, job_description: str) -> dict:
        """Analyze resume from text directly"""
        # Create a temporary workflow for text-only processing
        builder = StateGraph(MatchingState)
        builder.add_node("extract_keywords", self._extract_keywords)
        builder.add_node("compute_match_score", self._compute_match_score)
        builder.add_edge(START, "extract_keywords")
        builder.add_edge("extract_keywords", "compute_match_score")
        builder.add_edge("compute_match_score", END)
        text_workflow = builder.compile()

        initial_state = MatchingState(
            resume_text=resume_text,
            job_text=job_description,
            resume_keywords=[],
            job_keywords=[],
            missing_keywords=[],
            match_score=0.0,
            keyword_coverage=0.0,
            similarity_details=[],
            resume_bytes=b''
        )

        result = await text_workflow.ainvoke(initial_state)
        return self._format_result(result)

    def _format_result(self, result: MatchingState) -> dict:
        """Format the final result dictionary"""
        return {
            "match_score": result["match_score"],
            "keyword_coverage": result["keyword_coverage"],
            "similarity_details": result["similarity_details"],
            "missing_keywords": result["missing_keywords"],
            "resume_keywords": result["resume_keywords"],
            "job_keywords": result["job_keywords"]
        }

    def _parse_resume(self, state: MatchingState) -> dict:
        """Parse PDF resume to text"""
        resume_text = self.resume_parser.parse_pdf(state["resume_bytes"])
        return {"resume_text": resume_text}

    async def _extract_keywords(self, state: MatchingState) -> dict:
        """Extract keywords from resume and job description"""
        resume_keywords = await extract_keywords(state["resume_text"], self.llm)
        job_keywords = await extract_keywords(state["job_text"], self.llm)

        # Find missing keywords
        missing = list(set(job_keywords) - set(resume_keywords))

        return {
            "resume_keywords": resume_keywords,
            "job_keywords": job_keywords,
            "missing_keywords": missing
        }

    async def _compute_match_score(self, state: MatchingState) -> dict:
        """Compute match score between resume and job keywords"""
        resume_keywords = state["resume_keywords"]
        job_keywords = state["job_keywords"]

        if not resume_keywords or not job_keywords:
            return {
                "match_score": 0.0,
                "keyword_coverage": 0.0,
                "similarity_details": []
            }

        # Get embeddings - using await to fix the runtime warning
        resume_embeddings = await self.text_embedder.get_embeddings(resume_keywords)
        job_embeddings = await self.text_embedder.get_embeddings(job_keywords)

        # Compute similarity matrix
        similarity_matrix = torch.mm(resume_embeddings, job_embeddings.T)

        # Get best matches
        best_matches_job = torch.max(similarity_matrix, dim=0).values
        best_matches_resume = torch.max(similarity_matrix, dim=1).values

        # Calculate keyword coverage
        match_threshold = 0.7
        keyword_matches = [
            {"keyword": job_keywords[i], "match_score": float(score)}
            for i, score in enumerate(best_matches_job)
            if score > match_threshold
        ]
        keyword_coverage = len(keyword_matches) / len(job_keywords) if job_keywords else 0

        # Calculate overall score
        job_match_score = float(torch.mean(best_matches_job))
        resume_match_score = float(torch.mean(best_matches_resume))

        final_score = (
                0.6 * keyword_coverage +
                0.2 * job_match_score +
                0.2 * resume_match_score
        )

        return {
            "match_score": max(0.0, min(1.0, final_score)),
            "keyword_coverage": keyword_coverage,
            "similarity_details": keyword_matches
        }

    def _create_workflow(self) -> CompiledStateGraph:
        """Create workflow for resume matching from PDF bytes"""
        builder = StateGraph(MatchingState)

        # Add nodes
        builder.add_node("parse_resume", self._parse_resume)
        builder.add_node("extract_keywords", self._extract_keywords)
        builder.add_node("compute_match_score", self._compute_match_score)

        # Define flow
        builder.add_edge(START, "parse_resume")
        builder.add_edge("parse_resume", "extract_keywords")
        builder.add_edge("extract_keywords", "compute_match_score")
        builder.add_edge("compute_match_score", END)

        workflow = builder.compile()
        print(workflow.get_graph().draw_ascii())

        return workflow