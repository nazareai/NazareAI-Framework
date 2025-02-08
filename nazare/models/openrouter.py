"""
OpenRouter model implementation for the NazareAI Framework.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from loguru import logger
import openai
from openai import AsyncOpenAI

from nazare.core.model import Model, ModelConfig
from pydantic import ConfigDict, Field


class OpenRouterConfig(ModelConfig):
    """Configuration for OpenRouter models."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    api_base: Optional[str] = Field(None, description="Base URL for API calls")
    http_referer: str = Field(
        "https://github.com/nazareai/NazareAI-Framework",
        description="HTTP referer header for API calls"
    )
    http_user_agent: str = Field(
        "NazareAI/0.1.0",
        description="User agent header for API calls"
    )

    def __init__(self, **kwargs):
        """Initialize with defaults for OpenRouter."""
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://openrouter.ai/api/v1"
        super().__init__(**kwargs)


class OpenRouterModel(Model):
    """Implementation of OpenRouter model connectivity."""

    def __init__(self, config: OpenRouterConfig):
        super().__init__(config)
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the API client."""
        if self.config.api_key:
            # Set up default headers
            self.default_headers = {
                "HTTP-Referer": self.config.http_referer,
                "X-Title": "NazareAI Framework",
                "User-Agent": self.config.http_user_agent,
            }
            
            # Initialize AsyncOpenAI client
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                default_headers=self.default_headers
            )

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
        Generate a response using the OpenRouter API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional model parameters

        Returns:
            Generated text response
        """
        try:
            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": str(prompt)}],
                "temperature": temperature,
                **({"max_tokens": max_tokens} if max_tokens else {}),
                **({"stop": stop_sequences} if stop_sequences else {})
            }
            
            logger.debug(f"Making API request with params: {request_params}")
            
            response = await self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

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
        Stream a response using the OpenRouter API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional model parameters

        Yields:
            Generated text chunks
        """
        try:
            # Convert prompt to string if it's a dictionary
            if isinstance(prompt, dict):
                prompt_str = prompt.get("q", "") or prompt.get("query", "") or str(prompt)
            else:
                prompt_str = str(prompt)

            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt_str}],
                "temperature": temperature,
                "stream": True,
                **({"max_tokens": max_tokens} if max_tokens else {}),
                **({"stop": stop_sequences} if stop_sequences else {})
            }
            
            logger.debug(f"Making streaming API request with params: {request_params}")
            
            stream = await self.client.chat.completions.create(**request_params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise

    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using the OpenRouter API.

        Args:
            text: Input text or list of texts to embed
            **kwargs: Additional model parameters

        Returns:
            List of embeddings or list of lists for multiple inputs
        """
        try:
            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "input": text,
                **kwargs
            }
            
            logger.debug(f"Making embeddings request with params: {request_params}")
            
            response = await self.client.embeddings.create(**request_params)
            if isinstance(text, str):
                return response.data[0].embedding
            return [data.embedding for data in response.data]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 