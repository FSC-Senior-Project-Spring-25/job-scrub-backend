from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, Dict, Any, List, AsyncGenerator
import json
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


class ResponseFormat(Enum):
    """Enum for response format types"""
    RAW = "raw"
    JSON = "json"


@dataclass
class LLMResponse:
    """Container for LLM responses"""
    content: Union[str, Dict[str, Any]]
    raw_response: str
    success: bool
    error: Optional[str] = None


class LLM(ABC):
    """Base interface for LLM interactions"""

    @abstractmethod
    def __init__(
            self,
            model: str,
            temperature: float = 0.0,
            max_retries: int = 2
    ):
        """Initialize LLM interface

        Args:
            model: The LLM model to use
            temperature: Controls randomness in output (0.0 = deterministic)
            max_retries: Number of retry attempts for failed calls
        """
        load_dotenv()
        self.chat = None

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
    ) -> LLMResponse:
        """Parse the LLM response based on desired format

        Args:
            response_text: Raw response from the LLM
            format_type: Desired response format

        Returns:
            LLMResponse object containing parsed content
        """
        if format_type == ResponseFormat.RAW:
            return LLMResponse(
                content=response_text,
                raw_response=response_text,
                success=True
            )

        try:
            # Try to parse as JSON, cleaning up common formatting issues
            cleaned_text = response_text.strip()

            # Extract JSON content if it's wrapped in markdown code blocks
            if "```json" in cleaned_text:
                start = cleaned_text.find("```json") + 7
                end = cleaned_text.find("```", start)
                if end != -1:
                    cleaned_text = cleaned_text[start:end].strip()
            elif cleaned_text.startswith("```") and "```" in cleaned_text[3:]:
                start = cleaned_text.find("```") + 3
                end = cleaned_text.find("```", start)
                if end != -1:
                    cleaned_text = cleaned_text[start:end].strip()

            # Find the first { and last } to extract valid JSON if there's extra text
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')

            if first_brace != -1 and last_brace != -1:
                cleaned_text = cleaned_text[first_brace:last_brace + 1]

            json_content = json.loads(cleaned_text)
            return LLMResponse(
                content=json_content,
                raw_response=response_text,
                success=True
            )
        except json.JSONDecodeError as e:
            return LLMResponse(
                content={},
                raw_response=response_text,
                success=False,
                error=f"Failed to parse JSON response: {str(e)}"
            )

    async def agenerate(
            self,
            system_prompt: str,
            user_message: str,
            response_format: ResponseFormat = ResponseFormat.RAW
    ) -> LLMResponse:
        """Generate a response from the LLM

        Args:
            system_prompt: System instruction prompt
            user_message: User's input message
            response_format: Desired format for the response

        Returns:
            LLMResponse object containing the response
        """
        try:
            messages = self._create_messages(system_prompt, user_message, response_format)
            response = await self.chat.ainvoke(messages)
            return self._parse_response(response.content, response_format)
        except Exception as e:
            return LLMResponse(
                content="" if response_format == ResponseFormat.RAW else {},
                raw_response="",
                success=False,
                error=f"Generation failed: {str(e)}"
            )

    def generate(
            self,
            system_prompt: str,
            user_message: str,
            response_format: ResponseFormat = ResponseFormat.RAW
    ) -> LLMResponse:
        """Generate a response from the LLM

        Args:
            system_prompt: System instruction prompt
            user_message: User's input message
            response_format: Desired format for the response

        Returns:
            LLMResponse object containing the response
        """
        try:
            messages = self._create_messages(system_prompt, user_message, response_format)
            response = self.chat.invoke(messages)
            return self._parse_response(response.content, response_format)
        except Exception as e:
            return LLMResponse(
                content="" if response_format == ResponseFormat.RAW else {},
                raw_response="",
                success=False,
                error=f"Generation failed: {str(e)}"
            )

    async def generate_stream(
            self,
            system_prompt: str,
            user_message: str,
            response_format: ResponseFormat = ResponseFormat.RAW
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM

        Args:
            system_prompt: System instruction prompt
            user_message: User's input message
            response_format: Desired format for the response

        Yields:
            Chunks of the generated response
        """
        try:
            messages = self._create_messages(system_prompt, user_message, response_format)
            stream = self.chat.astream(messages)

            buffer = ""
            async for chunk in stream:
                if chunk.content:
                    if response_format == ResponseFormat.JSON:
                        # Buffer JSON content
                        buffer += chunk.content
                    else:
                        yield chunk.content

            # If JSON format, yield the complete buffered content
            if response_format == ResponseFormat.JSON and buffer:
                try:
                    json_content = json.loads(buffer)
                    yield json.dumps(json_content)
                except json.JSONDecodeError:
                    yield buffer

        except Exception as e:
            yield f"Error: {str(e)}"