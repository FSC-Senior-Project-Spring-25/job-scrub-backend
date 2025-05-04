from __future__ import annotations

import json
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


@dataclass
class LLMResponse:
    """Uniform wrapper for sync / async calls."""
    content: Union[str, BaseModel, Dict[str, Any]]
    raw_response: str
    success: bool
    error: Optional[str] = None


class LLM(ABC):
    """
    Base wrapper around a `BaseChatModel`.
    """

    @abstractmethod
    def __init__(
            self,
            model: str,
            temperature: float = 0.0,
            max_retries: int = 2,
    ):
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Sub‑classes must assign an actual chat model
        self.chat: BaseChatModel = None  # type: ignore

    @staticmethod
    def _create_messages(
            system_prompt: str, user_message: str
    ) -> List[BaseMessage]:
        """ Create messages for the LLM."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

    @staticmethod
    def _as_string(resp_content: Any) -> str:
        """Best‑effort serialization for `raw_response`."""
        if isinstance(resp_content, str):
            return resp_content
        if isinstance(resp_content, BaseModel):
            return resp_content.model_dump_json()
        try:
            return json.dumps(resp_content)
        except Exception:
            return str(resp_content)

    def generate(
            self,
            system_prompt: str,
            user_message: str,
            *,
            response_format: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            system_prompt (str): The system prompt to set the context.
            user_message (str): The user message to generate a response for.
            response_format (Optional[Type[BaseModel]]): The expected output schema.
        """
        try:
            msgs = self._create_messages(system_prompt, user_message)
            # Use `with_structured_output` if an output schema is provided
            chat = (
                self.chat.with_structured_output(response_format)
                if response_format
                else self.chat
            )
            resp = chat.invoke(msgs)
            content = resp.content if isinstance(resp, BaseMessage) else resp
            return LLMResponse(
                content=content,
                raw_response=self._as_string(content),
                success=True,
            )

        except Exception as exc:
            return LLMResponse(
                content={},
                raw_response="",
                success=False,
                error=f"Generation failed: {exc}",
            )

    async def agenerate(
            self,
            system_prompt: str,
            user_message: str,
            *,
            response_format: Optional[Union[typing.Dict, type]] = None,
    ) -> LLMResponse:
        """
        Async generate a response from the LLM.

        Args:
            system_prompt (str): The system prompt to set the context.
            user_message (str): The user message to generate a response for.
            response_format (Optional[Type[BaseModel]]): The expected output schema.
        """
        try:
            msgs = self._create_messages(system_prompt, user_message)
            # Use `with_structured_output` if an output schema is provided
            chat = (
                self.chat.with_structured_output(response_format)
                if response_format
                else self.chat
            )
            resp = await chat.ainvoke(msgs)
            print(f"[LLM] Response: {resp}", type(resp))
            content = resp.content if isinstance(resp, BaseMessage) else resp
            return LLMResponse(
                content=content,
                raw_response=self._as_string(content),
                success=True,
            )

        except Exception as exc:
            print(f"[LLM] Generation failed: {exc}")
            return LLMResponse(
                content={},
                raw_response="",
                success=False,
                error=f"Generation failed: {exc}",
            )

    async def generate_stream(
            self,
            system_prompt: str,
            user_message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response from the LLM in streaming mode.

        Args:
            system_prompt (str): The system prompt to set the context.
            user_message (str): The user message to generate a response for.
        """
        try:
            msgs = self._create_messages(system_prompt, user_message)
            async for chunk in self.chat.astream(msgs):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            yield f"Error: {exc}"
