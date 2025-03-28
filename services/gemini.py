from enum import Enum
from typing import Optional, Union, Dict, Any, List
import json
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


class ResponseFormat(Enum):
    """Enum for response format types"""
    RAW = "raw"
    JSON = "json"


@dataclass
class GeminiResponse:
    """Container for Gemini responses"""
    content: Union[str, Dict[str, Any]]
    raw_response: str
    success: bool
    error: Optional[str] = None


class GeminiLLM:
    """Base interface for Gemini LLM interactions"""

    def __init__(
            self,
            model: str = "gemini-2.0-flash",
            temperature: float = 0.0,
            max_tokens: Optional[int] = 512,
            max_retries: int = 2
    ):
        """Initialize Gemini LLM interface

        Args:
            model: The Gemini model to use
            temperature: Controls randomness in output (0.0 = deterministic)
            max_tokens: Maximum number of tokens in response
            max_retries: Number of retry attempts for failed calls
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )

    def _create_messages(
            self,
            system_prompt: str,
            user_message: str,
            response_format: ResponseFormat
    ) -> List[BaseMessage]:
        """Create formatted messages for the LLM

        Args:
            system_prompt: The system instruction prompt
            user_message: The user's input message
            response_format: Desired format for the response

        Returns:
            List of formatted messages
        """
        format_instruction = ""
        if response_format == ResponseFormat.JSON:
            format_instruction = "\nProvide your response as a valid JSON object."

        return [
            SystemMessage(content=f"{system_prompt}{format_instruction}"),
            HumanMessage(content=user_message)
        ]

    def _parse_response(
            self,
            response_text: str,
            format_type: ResponseFormat
    ) -> GeminiResponse:
        """Parse the LLM response based on desired format

        Args:
            response_text: Raw response from the LLM
            format_type: Desired response format

        Returns:
            GeminiResponse object containing parsed content
        """
        if format_type == ResponseFormat.RAW:
            return GeminiResponse(
                content=response_text,
                raw_response=response_text,
                success=True
            )

        try:
            # Try to parse as JSON, cleaning up common formatting issues
            cleaned_text = response_text.strip()
            # Remove markdown code block markers if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            json_content = json.loads(cleaned_text.strip())
            return GeminiResponse(
                content=json_content,
                raw_response=response_text,
                success=True
            )
        except json.JSONDecodeError as e:
            return GeminiResponse(
                content={},
                raw_response=response_text,
                success=False,
                error=f"Failed to parse JSON response: {str(e)}"
            )

    async def generate(
            self,
            system_prompt: str,
            user_message: str,
            response_format: ResponseFormat = ResponseFormat.RAW
    ) -> GeminiResponse:
        """Generate a response from Gemini

        Args:
            system_prompt: System instruction prompt
            user_message: User's input message
            response_format: Desired format for the response

        Returns:
            GeminiResponse object containing the response
        """
        try:
            messages = self._create_messages(system_prompt, user_message, response_format)
            response = await self.llm.ainvoke(messages)
            return self._parse_response(response.content, response_format)
        except Exception as e:
            return GeminiResponse(
                content="" if response_format == ResponseFormat.RAW else {},
                raw_response="",
                success=False,
                error=f"Generation failed: {str(e)}"
            )
