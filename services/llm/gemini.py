from langchain_google_genai import ChatGoogleGenerativeAI

from services.llm.base.llm import LLM


class GeminiLLM(LLM):
    """Interface for Gemini LLM interactions"""

    def __init__(
            self,
            model: str = "gemini-2.0-flash-lite",
            temperature: float = 0.0,
            max_retries: int = 2
    ):
        """Initialize Gemini LLM interface

        Args:
            model: The Gemini model to use
            temperature: Controls randomness in output (0.0 = deterministic)
            max_retries: Number of retry attempts for failed calls
        """
        super().__init__(model, temperature, max_retries)
        self.chat = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )
