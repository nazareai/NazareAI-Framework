"""
Base model implementation for the NazareAI Framework.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    """Configuration for AI models."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(..., description="Name of the model")
    api_key: Optional[str] = Field(None, description="API key for the model service")
    api_base: Optional[str] = Field(None, description="Base URL for API calls")
    timeout: float = Field(30.0, description="Timeout for API calls in seconds")
    max_retries: int = Field(3, description="Maximum number of retries for failed calls")


class Model(ABC):
    """Base class for all AI models in the framework."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the model configuration."""
        if not self.config.model_name:
            raise ValueError("Model name must be specified")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the model.

        Args:
            prompt: The input prompt for the model
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional model-specific parameters

        Yields:
            Generated text chunks
        """
        pass

    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the input text.

        Args:
            text: Input text or list of texts to embed
            **kwargs: Additional model-specific parameters

        Returns:
            List of embeddings or list of lists for multiple inputs
        """
        pass

    def prepare_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Prepare a prompt string from messages and system prompt.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend

        Returns:
            Formatted prompt string
        """
        formatted_messages = []
        
        if system_prompt:
            formatted_messages.append(f"System: {system_prompt}\n")
        
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_messages.append(f"{role}: {content}")
        
        return "\n".join(formatted_messages)

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 