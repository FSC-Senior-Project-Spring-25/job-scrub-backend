from fastapi import HTTPException

from models.job_report import JobReport
from services.llm.base.llm import LLM
from services.llm.groq import GroqLLM


async def generate_job_report(
        content: str,
        llm: LLM = GroqLLM(),
) -> JobReport:
    """
    Analyze unstructured job content and return a structured JobReport.
    """

    system_prompt = """
    You are a job‑posting analyzer. Using the schema below, return ONLY a
    valid JSON object that conforms exactly to the fields and types.

    {schema}

    IMPORTANT:
    - No markdown fences or explanations.
    - All required fields must be present.
    """.format(schema=JobReport.model_json_schema())

    resp = await llm.agenerate(
        system_prompt=system_prompt,
        user_message=content,
        response_format=JobReport,
    )

    if not resp.success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze job content: {resp.error}",
        )

    if not isinstance(resp, JobReport):
        raise HTTPException(status_code=500, detail="Failed to parse job report")

    return resp.content
