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
        initial_state = MatchingState(
            resume_text="",
            job_text=job_description,
            missing_keywords=[],
            resume_keywords=[],
            job_keywords=[],
            match_score=0.0,
            keyword_coverage=0.0,
            similarity_details=[],
            resume_bytes=resume_bytes
        )

        result = await self.workflow.ainvoke(initial_state)

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
        """Extract keywords from resume using Gemini"""
        resume_keywords = await extract_keywords(state["resume_text"], self.llm)
        job_keywords = await extract_keywords(state["job_text"], self.llm)
        return {
            "resume_keywords": resume_keywords,
            "job_keywords": job_keywords
        }

    def _compute_match_score(self, state: MatchingState) -> dict:
        """
        Compute comprehensive match score between resume and job keywords

        Args:
            state: Current matching state containing resume and job keywords
        Returns:
            Dictionary with match scores and details
        """
        resume_keywords = state["resume_keywords"]
        job_keywords = state["job_keywords"]
        if not resume_keywords or not job_keywords:
            return {
                "match_score": 0.0,
                "keyword_coverage": 0.0,
                "similarity_details": []
            }

        # Get embeddings
        resume_embeddings = self.text_embedder.get_embeddings(resume_keywords)
        job_embeddings = self.text_embedder.get_embeddings(job_keywords)

        # Compute similarity matrix
        similarity_matrix = torch.mm(resume_embeddings, job_embeddings.T)

        # Get best matches in both directions
        best_matches_job = torch.max(similarity_matrix, dim=0).values  # Best resume match for each job keyword
        best_matches_resume = torch.max(similarity_matrix, dim=1).values  # Best job match for each resume keyword

        # Calculate keyword coverage (how many job keywords are well-matched)
        keyword_matches = [(job_keywords[i], float(score))
                           for i, score in enumerate(best_matches_job)
                           if score > 0.7]  # threshold for good match
        keyword_coverage = len(keyword_matches) / len(job_keywords)

        # Calculate bidirectional matching score
        job_match_score = float(torch.mean(best_matches_job))
        resume_match_score = float(torch.mean(best_matches_resume))

        # Compute final score with emphasis on job keyword coverage
        final_score = (
                0.6 * keyword_coverage +  # Coverage of job keywords
                0.2 * job_match_score +  # How well resume matches job requirements
                0.2 * resume_match_score  # How well job matches resume skills
        )

        return {
            "match_score": max(0.0, min(1.0, final_score)),
            "keyword_coverage": keyword_coverage,
            "missing_keywords": [set(job_keywords) - set(resume_keywords)],
            "similarity_details": [
                {"keyword": kw, "match_score": score}
                for kw, score in keyword_matches
            ]
        }

    def _create_workflow(self) -> CompiledStateGraph:
        """Compile workflow for resume matching"""
        builder = StateGraph(MatchingState)

        # Add nodes
        builder.add_node("parse_resume", self._parse_resume)
        builder.add_node("extract_keywords", self._extract_keywords)
        builder.add_node("compute_match_score", self._compute_match_score)

        # Define the workflow
        builder.add_edge(START, "parse_resume")
        builder.add_edge("parse_resume", "extract_keywords")
        builder.add_edge("extract_keywords", "compute_match_score")
        builder.add_edge("compute_match_score", END)

        workflow = builder.compile()
        print(workflow.get_graph().draw_ascii())

        return workflow
