"""
Basic example usage of the NazareAI Framework showing different ways to use models and plugins.
"""

import asyncio
from pathlib import Path
from typing import Optional

from nazare.core.agent import Agent, AgentConfig
from nazare.models.openrouter import OpenRouterConfig, OpenRouterModel
from nazare.models.replicate import ReplicateConfig, ReplicateModel
from nazare.plugins.serpapi import SerpApiConfig, SerpApiPlugin
from nazare.utils.config import load_settings


async def create_basic_agent() -> Agent:
    """
    Create a basic agent with minimal configuration.
    This is the simplest way to use the framework.
    """
    settings = load_settings()

    # Configure model
    model_config = OpenRouterConfig(
        model_name=settings.default_model,
        api_key=settings.openrouter_api_key,
        api_base=settings.openrouter_api_base
    )
    model = OpenRouterModel(model_config)

    # Configure agent
    agent_config = AgentConfig(
        name="basic-assistant",
        description="A simple AI assistant",
        model_settings=model_config,
        prompt_library_path="prompts"
    )

    # Create and initialize agent
    agent = Agent(config=agent_config, model=model)
    await agent.initialize()
    return agent


async def create_advanced_agent(use_replicate: bool = False) -> Agent:
    """
    Create an advanced agent with plugins and custom configuration.
    This shows how to use more framework features.

    Args:
        use_replicate: Whether to use Replicate model instead of OpenRouter
    """
    settings = load_settings()

    # Configure model based on preference and availability
    if use_replicate and settings.replicate_api_token and settings.replicate_api_token != "your-replicate-api-token-here":
        # Configure Replicate model
        model_config = ReplicateConfig(
            model_name="meta/llama-2-70b-chat",
            version="02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            api_key=settings.replicate_api_token,
            input_params={
                "system_prompt": "You are a helpful AI assistant.",
                "max_new_tokens": 500,
                "temperature": 0.7
            }
        )
        model = ReplicateModel(model_config)
        print("Using Replicate model")
    else:
        # Use OpenRouter model
        model_config = OpenRouterConfig(
            model_name="anthropic/claude-3.5-sonnet:beta",  # Use Claude 3.5 Sonnet
            api_key=settings.openrouter_api_key,
            api_base=settings.openrouter_api_base,
            timeout=settings.timeout
        )
        model = OpenRouterModel(model_config)
        print("Using OpenRouter model")

    # Configure search plugin
    serpapi_config = SerpApiConfig(
        api_key=settings.serpapi_api_key,
        timeout=settings.timeout
    )
    search_plugin = SerpApiPlugin(config=serpapi_config.model_dump())

    # Ensure prompts directory exists
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)

    # Configure agent with more options
    agent_config = AgentConfig(
        name="advanced-assistant",
        description="An AI assistant with search capabilities",
        model_settings=model_config,
        prompt_library_path=str(prompts_dir),
        debug=settings.debug,
        timeout=settings.timeout,
        max_retries=settings.max_retries
    )

    # Create and initialize agent with plugin
    agent = Agent(config=agent_config, model=model, plugins=[search_plugin])
    await agent.initialize()
    return agent


async def process_query(
    agent: Agent,
    query: str,
    *,
    template_name: Optional[str] = None,
    template_version: Optional[str] = None,
    template_params: Optional[dict] = None,
    stream: bool = True
) -> None:
    """
    Process a query through an agent.
    Shows different ways to use templates and streaming.
    """
    try:
        print(f"\nQuery: {query}")
        if template_name:
            print(f"Using template: {template_name} v{template_version or 'latest'}")

        # Ensure query is included in template parameters
        params = template_params or {}
        params["query"] = query

        if stream:
            print("\nResponse:")
            async for chunk in agent.process_stream(
                query,
                template=template_name,
                template_version=template_version,
                **params
            ):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            response = await agent.process(
                query,
                template=template_name,
                template_version=template_version,
                **params
            )
            print(f"\nResponse:\n{response}\n")

    except Exception as e:
        print(f"Error processing query: {e}")


async def main():
    """
    Run examples showing different ways to use the framework.
    """

    # Example 2: Advanced usage with OpenRouter
    print("\n=== Example 2: Advanced Agent (OpenRouter) ===")
    advanced_agent = await create_advanced_agent(use_replicate=False)
    try:
        # Query with search plugin
        await process_query(
            advanced_agent,
            "Who is 0xroyce369?",
            template_name="research",
            template_version="1.0.0",
            template_params={
                "depth": "detailed",
                "format": "markdown",
                "context": ""
            }
        )
    finally:
        await advanced_agent.cleanup()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 