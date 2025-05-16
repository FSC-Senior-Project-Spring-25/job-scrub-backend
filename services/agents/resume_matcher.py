"""ResumeMatchingAgent – compares a candidate résumé to a job description.
Adds conversation-history awareness and a pre‑processing node that extracts job
text from the latest relevant chat message if none was provided explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import torch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from models.chat import Message
from services.agents.base.agent import Agent, AgentResponse
from services.agents.tools.extract_keywords import extract_keywords
from services.llm.base.llm import LLM
from services.llm.gemini import GeminiLLM
from services.text_embedder import TextEmbedder


class MatchingState(TypedDict):
    prompt: str
    conversation_history: List[Message]
    job_text: str
    resume_keywords: List[str]
    job_keywords: List[str]
    missing_keywords: List[str]
    match_score: float
    keyword_coverage: float
    similarity_details: List[Dict[str, Any]]


class ResumeMatchingAgent(Agent):
    def __init__(
            self,
            embedder: TextEmbedder,
            resume_text: str,
            llm: LLM = GeminiLLM(),
    ):
        self.text_embedder = embedder
        self.resume_text = resume_text
        super().__init__(llm)

    async def invoke(
            self,
            prompt: str,
            history: List[Message],
    ) -> AgentResponse:
        init_state: MatchingState = {
            "prompt": prompt or "",
            "job_text": "",
            "conversation_history": history or [],
            "resume_keywords": [],
            "job_keywords": [],
            "missing_keywords": [],
            "match_score": 0.0,
            "keyword_coverage": 0.0,
            "similarity_details": [],
        }
        result = await self.workflow.ainvoke(init_state)
        return self._format_response(result)

    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        return (
            "You are a resume matching agent. Your task is to compare a candidate's "
            "resume with a job description and provide a match score based on keyword "
            "coverage and similarity."
        )

    def _create_workflow(self) -> CompiledStateGraph:
        """Create workflow for resume matching from PDF bytes"""
        builder = StateGraph(MatchingState)

        # Add nodes
        builder.add_node("fetch_job_text", self._fetch_job_text)
        builder.add_node("extract_keywords", self._extract_keywords)
        builder.add_node("compute_match_score", self._compute_match_score)

        # Define flow
        builder.add_edge(START, "fetch_job_text")
        builder.add_edge("fetch_job_text", "extract_keywords")
        builder.add_edge("extract_keywords", "compute_match_score")
        builder.add_edge("compute_match_score", END)

        workflow = builder.compile()
        return workflow

    def _extract_answer(self, result: MatchingState) -> dict:
        """Format the final result dictionary"""
        return {
            "match_score": result["match_score"],
            "keyword_coverage": result["keyword_coverage"],
            "similarity_details": result["similarity_details"],
            "missing_keywords": result["missing_keywords"],
            "resume_keywords": result["resume_keywords"],
            "job_keywords": result["job_keywords"]
        }

    async def _fetch_job_text(self, state: MatchingState) -> Dict[str, Any]:
        """Use LLM to identify the most recent job description from conversation history."""
        history = state.get("conversation_history", [])
        prompt = state.get("prompt", "")

        # Use LLM to extract job text from conversation history
        try:
            # Prepare context from most recent messages (limit to avoid token issues)
            recent_messages = history[-5:]
            context = prompt + "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])

            # Create prompt for job text extraction
            extraction_prompt = f"""
            Extract the most recent job description from the conversation below.
            Only return the job description text itself without any commentary.
            """

            # Query LLM to extract job text
            job_text = await self.llm.agenerate(extraction_prompt, context)
            job_text = job_text.content.strip()
            # Return extracted job text if substantial
            if len(job_text) > 50:
                return {"job_text": job_text}
        except Exception as e:
            print(f"Error extracting job text with LLM: {str(e)}")

        # Fall back to original heuristic if LLM extraction fails
        for msg in reversed(history):
            if msg.role == "user" and len(msg.content) > 200:
                return {"job_text": msg.content}

        return {}

    async def _extract_keywords(self, state: MatchingState) -> Dict[str, Any]:
        job_text = state["job_text"]
        resume_kw = await extract_keywords(self.resume_text, self.llm)
        job_kw: List[str] = []
        if job_text:
            job_kw = await extract_keywords(job_text, self.llm)
        missing = list(set(job_kw) - set(resume_kw))
        return {
            "resume_keywords": resume_kw,
            "job_keywords": job_kw,
            "missing_keywords": missing,
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
