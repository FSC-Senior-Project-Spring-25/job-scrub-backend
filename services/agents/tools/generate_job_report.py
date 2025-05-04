from fastapi import HTTPException

from models.job_report import JobReport
from services.llm.base.llm import ResponseFormat, LLM
from services.llm.groq import GroqLLM


async def generate_job_report(
        content: str,
        llm: LLM = GroqLLM(),
) -> JobReport:
    """
    Analyze unstructured job content and generate a structured JobReport using LLM

    Args:
        content: Raw job description text
        llm: LLM instance for generating structured data

    Returns:
        Structured JobReport from the content analysis
    """
    # Get JobReport schema details
    job_report_schema = JobReport.model_json_schema()

    system_prompt = f"""
    You are a job posting analyzer. Extract structured information from the provided job description.

    IMPORTANT FORMATTING REQUIREMENTS:
    - Return ONLY a valid JSON object with no additional text, explanations, or markdown
    - No prefixes, suffixes, or code blocks
    - The output must be parseable by Python's json.loads()
    - Do NOT truncate any fields
    - Do NOT omit any required fields

    SCHEMA DEFINITION:
    - title: string (job title)
    - company: string (company name)
    - url: string (job posting URL or empty string if not found)
    - description: string (full job description, include the complete text not including title, date, or text not relevant to the job)
    - jobType: string (MUST be exactly one of: "fulltime", "parttime", "internship", "contract", "volunteer")
    - skills: array of strings (required skills)
    - location: object with:
      - address: string (location name/address)
      - lat: float (latitude, use 0 if unknown)
      - lon: float (longitude, use 0 if unknown)
    - locationType: string (MUST be exactly one of: "remote", "onsite", "hybrid")
    - benefits: array of strings (job benefits)
    - salary: string (salary information or null if not specified)
    
    SCHEMA:
    {job_report_schema}

    Output JSON format example:
    {{"title": "Software Engineer", "company": "Example Inc", "url": "https://example.com", "description": "Full job description...", "jobType": "fulltime", "skills": ["Python", "FastAPI"], "location": {{"address": "Remote USA", "lat": 0, "lon": 0}}, "locationType": "remote", "benefits": ["Health Insurance", "401k"], "salary": "$100,000-$120,000"}}
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
