from services.gemini import GeminiLLM, ResponseFormat


async def extract_keywords(
        text: str,
        llm: GeminiLLM,
) -> list[str]:
    """Extract technical keywords from text using Gemini"""
    system_prompt = """You are a technical keyword extraction assistant. 
    Extract technical keywords from the job description.
    Valid keywords are technical terms or tools that are relevant to the job description.
    DO NOT include any personal information or specific project details.
    DO NOT include any punctuation or special characters.
    DO NOT include any generic terms or nouns like "software" or "tools".  
    The extracted keywords must be be specific and a valid technical term.
    
    Return ONLY a JSON object with a single 'keywords' field containing an array of strings.
    Example: {"keywords": ["Python", "AWS", "Docker"]}        
     """

    human_prompt = f"Extract keywords from the following: {text}"

    response = await llm.generate(
        system_prompt=system_prompt,
        user_message=human_prompt,
        response_format=ResponseFormat.JSON
    )

    if not response.success:
        print(f"Failed to extract keywords: {response.error}")
        return []

    try:
        if isinstance(response.content, dict):
            return response.content.get("keywords", [])
        return []
    except Exception as e:
        print(f"Failed to process keywords: {e}")
        return []