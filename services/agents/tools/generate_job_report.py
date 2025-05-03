from fastapi import HTTPException

from models.job_report import JobReport
from services.gemini import ResponseFormat, GeminiLLM


async def generate_job_report(
        content: str,
        llm: GeminiLLM = GeminiLLM(),
) -> JobReport:
    """
    Analyze unstructured job content and generate a structured JobReport using LLM

    Args:
        content: Raw job description text
        llm: LLM instance for generating structured data

    Returns:
        Structured JobReport from the content analysis
    """
    system_prompt = """
    You are a job posting analyzer. Extract structured information from the provided job description.
    Return a valid JSON object that matches the JobReport schema with these fields:
    - title: string (job title)
    - company: string (company name)
    - url: string (job posting URL or empty string if not found)
    - description: string (full job description)
    - jobType: string (one of: "fulltime", "parttime", "internship", "contract", "volunteer")
    - skills: array of strings (required skills)
    - location: object with:
      - address: string (location name/address)
      - lat: float (latitude, use 0 if unknown)
      - lon: float (longitude, use 0 if unknown)
    - locationType: string (one of: "remote", "onsite", "hybrid")
    - benefits: array of strings (job benefits)
    - salary: string (salary information or null if not specified)
    """

    # Get structured analysis from Gemini
    response = await llm.agenerate(
        system_prompt=system_prompt,
        user_message=content,
        response_format=ResponseFormat.JSON
    )

    if not response.success:
        raise HTTPException(status_code=500, detail=f"Failed to analyze job content: {response.error}")

    try:
        # Convert the JSON response to a JobReport model
        job_data = response.content

        # Ensure location object exists with required fields
        if "location" not in job_data or not isinstance(job_data["location"], dict):
            job_data["location"] = {"address": "", "lat": 0.0, "lon": 0.0}
        else:
            loc = job_data["location"]
            if "address" not in loc: loc["address"] = ""
            if "lat" not in loc: loc["lat"] = 0.0
            if "lon" not in loc: loc["lon"] = 0.0

        # Create and validate JobReport from the JSON
        job_report = JobReport.model_validate(job_data)
        return job_report

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid job report structure: {str(e)}")