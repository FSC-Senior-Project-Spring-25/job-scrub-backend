from langchain_groq import ChatGroq

from services.llm.base.llm import LLM


class GroqLLM(LLM):
    """Interface for Groq LLM interactions"""

    def __init__(
            self,
            model: str = "gemma2-9b-it",
            temperature: float = 0.0,
            max_retries: int = 2
    ):
        """Initialize Groq LLM interface

        Args:
            model: The Groq model to use
            temperature: Controls randomness in output (0.0 = deterministic)
            max_retries: Number of retry attempts for failed calls
        """
        super().__init__(model, temperature, max_retries)
        self.chat = ChatGroq(
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )