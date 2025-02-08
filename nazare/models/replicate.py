"""
Replicate model implementation for the NazareAI Framework.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
import replicate

from nazare.core.model import Model, ModelConfig
from pydantic import ConfigDict, Field


class ReplicateConfig(ModelConfig):
    """Configuration for Replicate models."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    version: str = Field(..., description="Model version hash")
    input_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default input parameters for the model"
    )
    stream_interval: float = Field(
        0.1,
        description="Interval in seconds between stream checks"
    )


class ReplicateModel(Model):
    """Implementation of Replicate model connectivity."""

    def __init__(self, config: ReplicateConfig):
        super().__init__(config)
        self.client = replicate.Client(api_token=config.api_key)

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
        Generate a response using the Replicate API.

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
            # Prepare input parameters
            input_params = {
                **self.config.input_params,
                "prompt": prompt,
                "temperature": temperature,
                **({"max_length": max_tokens} if max_tokens else {}),
                **({"stop_sequences": stop_sequences} if stop_sequences else {}),
                **kwargs
            }

            # Run the model
            output = await asyncio.to_thread(
                self.client.run,
                f"{self.config.model_name}:{self.config.version}",
                input=input_params
            )

            # Handle different output formats
            if isinstance(output, list):
                return "".join(str(chunk) for chunk in output)
            return str(output)

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
        Stream a response using the Replicate API.

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
            # Prepare input parameters
            input_params = {
                **self.config.input_params,
                "prompt": prompt,
                "temperature": temperature,
                **({"max_length": max_tokens} if max_tokens else {}),
                **({"stop_sequences": stop_sequences} if stop_sequences else {}),
                **kwargs
            }

            # Run the model
            prediction = await asyncio.to_thread(
                self.client.create_prediction,
                f"{self.config.model_name}:{self.config.version}",
                input=input_params
            )

            # Stream the output
            while prediction.status in ["starting", "processing"]:
                if prediction.output:
                    if isinstance(prediction.output, list):
                        for chunk in prediction.output:
                            yield str(chunk)
                    else:
                        yield str(prediction.output)
                
                await asyncio.sleep(self.config.stream_interval)
                prediction.reload()

            # Yield any remaining output
            if prediction.output and prediction.status == "succeeded":
                if isinstance(prediction.output, list):
                    for chunk in prediction.output[len(prediction.output):]:
                        yield str(chunk)
                else:
                    yield str(prediction.output)

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise

    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using the Replicate API.

        Args:
            text: Input text or list of texts to embed
            **kwargs: Additional model parameters

        Returns:
            List of embeddings or list of lists for multiple inputs
        """
        try:
            # Prepare input
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text

            # Process each text
            embeddings = []
            for t in texts:
                input_params = {
                    **self.config.input_params,
                    "text": t,
                    **kwargs
                }

                # Run the model
                output = await asyncio.to_thread(
                    self.client.run,
                    f"{self.config.model_name}:{self.config.version}",
                    input=input_params
                )

                embeddings.append(output)

            return embeddings[0] if isinstance(text, str) else embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise 