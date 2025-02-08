"""
Core agent implementation for the NazareAI Framework.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import uuid

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from nazare.core.model import Model, ModelConfig
from nazare.core.plugin import Plugin, PluginManager
from nazare.core.prompt import PromptLibrary, PromptTemplate


class AgentConfig(BaseModel):
    """Configuration for AI agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Name of the agent")
    description: str = Field("", description="Description of the agent")
    model_settings: ModelConfig = Field(..., description="Configuration for the agent's model")
    prompt_library_path: str = Field("prompts", description="Path to prompt library storage")
    max_retries: int = Field(3, description="Maximum number of retries for failed operations")
    timeout: float = Field(30.0, description="Timeout for operations in seconds")
    debug: bool = Field(False, description="Enable debug mode")


class AgentContext:
    """Context for managing agent state during conversations."""

    def __init__(self):
        self.conversation_id = str(uuid.uuid4())
        self.messages: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {}
        self.plugin_data: Dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history, optionally limited to last N messages."""
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:]

    def clear(self) -> None:
        """Clear the conversation history and metadata."""
        self.messages.clear()
        self.metadata.clear()
        self.plugin_data.clear()


class Agent:
    """Main agent class that orchestrates models, plugins, and prompts."""

    def __init__(
        self,
        config: AgentConfig,
        model: Optional[Model] = None,
        plugins: Optional[List[Plugin]] = None
    ):
        self.config = config
        self.model = model or self._create_model()
        self.plugin_manager = PluginManager()
        self.prompt_library = PromptLibrary(config.prompt_library_path)
        self.context = AgentContext()

        # Register plugins if provided
        if plugins:
            for plugin in plugins:
                self.plugin_manager.register(plugin)

        # Configure logging based on debug setting
        if self.config.debug:
            logger.enable("nazare")
            logger.level("DEBUG")
        else:
            logger.disable("nazare")
            logger.level("INFO")

    def _create_model(self) -> Model:
        """Create a model instance from configuration."""
        from nazare.models.openrouter import OpenRouterModel
        return OpenRouterModel(self.config.model_settings)

    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        logger.info(f"Initializing agent: {self.config.name}")
        
        # Initialize plugins
        await self.plugin_manager.initialize_all()
        logger.debug("Plugins initialized")

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        logger.info(f"Cleaning up agent: {self.config.name}")
        
        # Cleanup plugins
        await self.plugin_manager.cleanup_all()
        logger.debug("Plugins cleaned up")

    async def process(
        self,
        input_data: Union[str, Dict[str, Any]],
        template_name: Optional[str] = None,
        template_version: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Process input through the agent's pipeline.

        Args:
            input_data: Input string or data dictionary
            template_name: Optional name of prompt template to use
            template_version: Optional version of prompt template
            **kwargs: Additional keyword arguments for template formatting

        Returns:
            Processed output
        """
        try:
            # Convert input to string if needed
            if isinstance(input_data, dict):
                # Extract the actual query content from the dictionary
                if "q" in input_data:
                    input_str = input_data["q"]
                elif "query" in input_data:
                    input_str = input_data["query"]
                else:
                    input_str = str(input_data)
            else:
                input_str = input_data

            # Format prompt if template is specified
            if template_name:
                template = self.prompt_library.get_template(
                    template_name,
                    version=template_version
                )
                # Ensure query is included in template parameters
                template_params = {**kwargs}
                if "query" not in template_params:
                    template_params["query"] = input_str
                input_str = template.format(**template_params)

            # Add user message to context
            self.context.add_message("user", input_str)

            # Pre-process through plugins
            processed_input = input_str
            for plugin_name in self.plugin_manager._plugins:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                processed_input = await plugin.pre_process(processed_input)

            # Generate response from model
            logger.debug(f"Generating response for input: {processed_input}")
            # Ensure we're passing a string to the model
            if isinstance(processed_input, dict):
                processed_input = processed_input.get("q", str(processed_input))
            response = await self.model.generate(processed_input)

            # Post-process through plugins
            processed_output = response
            for plugin_name in self.plugin_manager._plugins:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                processed_output = await plugin.post_process(processed_output)

            # Add assistant message to context
            self.context.add_message("assistant", processed_output)

            return processed_output

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise

    async def process_stream(
        self,
        input_data: Union[str, Dict[str, Any]],
        template: Optional[str] = None,
        template_version: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Process input data and generate a streaming response."""
        try:
            # If using a template, format the input first
            if template:
                template_obj = self.prompt_library.get_template(template, template_version)
                # Ensure query is included in template parameters
                template_params = {**kwargs}
                
                # Handle input_data and ensure query is properly set
                if isinstance(input_data, dict):
                    query = input_data.get("q") or input_data.get("query") or str(input_data)
                else:
                    query = str(input_data)
                
                template_params["query"] = query
                
                # Add default values for optional parameters if not already set
                if "context" not in template_params:
                    template_params["context"] = ""
                if "depth" not in template_params:
                    template_params["depth"] = template_params.get("depth", "detailed")
                if "format" not in template_params:
                    template_params["format"] = template_params.get("format", "markdown")
                
                # Format using template with all parameters
                processed_input = template_obj.format(**template_params)
            else:
                # For non-template input, ensure we have a string
                if isinstance(input_data, dict):
                    processed_input = input_data.get("q") or input_data.get("query") or str(input_data)
                else:
                    processed_input = str(input_data)

            # Log the input being sent to the model
            logger.debug(f"Generating streaming response for input: {processed_input}")

            # Add user message to context
            self.context.add_message("user", processed_input)

            # Pre-process through plugins
            for plugin_name in self.plugin_manager._plugins:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                processed_input = await plugin.pre_process(processed_input)

            # Extract model-specific parameters from kwargs
            model_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in ["temperature", "max_tokens", "stop_sequences"]
            }

            # Generate streaming response
            response_text = ""
            async for chunk in self.model.generate_stream(processed_input, **model_kwargs):
                # Post-process chunk through plugins
                for plugin_name in self.plugin_manager._plugins:
                    plugin = self.plugin_manager.get_plugin(plugin_name)
                    chunk = await plugin.post_process(chunk)
                response_text += chunk
                yield chunk

            # Add complete assistant message to context
            self.context.add_message("assistant", response_text)

        except Exception as e:
            logger.error(f"Error processing streaming input: {e}")
            raise

    def add_plugin(self, plugin: Plugin) -> None:
        """
        Add a plugin to the agent.

        Args:
            plugin: Plugin instance to add
        """
        self.plugin_manager.register(plugin)

    def remove_plugin(self, plugin_name: str) -> None:
        """
        Remove a plugin from the agent.

        Args:
            plugin_name: Name of the plugin to remove
        """
        self.plugin_manager.unregister(plugin_name)

    def get_plugin(self, plugin_name: str) -> Plugin:
        """
        Get a plugin by name.

        Args:
            plugin_name: Name of the plugin to retrieve

        Returns:
            Plugin instance
        """
        return self.plugin_manager.get_plugin(plugin_name)

    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup() 