from typing import List
from pydantic import BaseModel
from services.llm.base.llm import LLM
from services.llm.gemini import GeminiLLM


class KeywordList(BaseModel):
    keywords: List[str]


async def extract_keywords(
    text: str,
    llm: LLM = GeminiLLM(),
) -> list[str]:
    """Extract technical keywords from plain text."""
    system_prompt = """
    You are a technical keyword‑extraction assistant.
    Extract only *specific* technical terms or tools relevant to the job description.
    Return exactly: {"keywords": ["term1", "term2", ...]}
    """

    response = await llm.agenerate(
        system_prompt=system_prompt,
        user_message=f"Extract keywords from: {text}",
        response_format=KeywordList,
    )

    if not response.success:
        print(f"Failed to extract keywords: {response.error}")
        return []

    return response.content.keywords
