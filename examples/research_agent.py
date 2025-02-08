"""
Research Agent implementation using the NazareAI Framework.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional

from nazare.core.agent import Agent, AgentConfig
from nazare.models.openrouter import OpenRouterConfig, OpenRouterModel
from nazare.plugins.serpapi import SerpApiConfig, SerpApiPlugin
from nazare.utils.config import load_settings


class ResearchAgent:
    """A specialized agent for conducting research."""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self.settings = load_settings()
        
        # Ensure prompts directory exists
        self.prompts_dir = Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Create research prompts directory if it doesn't exist
        self.research_prompts_dir = self.prompts_dir / "research"
        self.research_prompts_dir.mkdir(exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the research agent with necessary components."""
        # Configure the model (using Claude-3 for best research capabilities)
        model_config = OpenRouterConfig(
            model_name=self.settings.default_model,
            api_key=self.settings.openrouter_api_key,
            api_base=self.settings.openrouter_api_base,
            timeout=self.settings.timeout,
            max_retries=self.settings.max_retries
        )
        model = OpenRouterModel(model_config)

        # Configure the search plugin
        serpapi_config = SerpApiConfig(
            api_key=self.settings.serpapi_api_key,
            timeout=self.settings.timeout,
            max_results=10  # Get more results for comprehensive research
        )
        search_plugin = SerpApiPlugin(config=serpapi_config.model_dump())

        # Configure the agent
        agent_config = AgentConfig(
            name="research-assistant",
            description="An agent that conducts thorough research and analysis",
            model_settings=model_config,
            prompt_library_path=str(self.prompts_dir),
            debug=self.settings.debug,
            timeout=self.settings.timeout,
            max_retries=self.settings.max_retries
        )

        # Create and initialize agent with plugin
        self.agent = Agent(config=agent_config, model=model, plugins=[search_plugin])
        await self.agent.initialize()

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.agent:
            await self.agent.cleanup()

    async def research(
        self,
        topic: str,
        *,
        depth: str = "detailed",
        format: str = "markdown",
        stream: bool = True
    ) -> None:
        """
        Conduct research on a given topic.

        Args:
            topic: The research topic to investigate
            depth: Research depth (basic/detailed/comprehensive)
            format: Output format (markdown/academic/report)
            stream: Whether to stream the output
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        try:
            print(f"\nResearching topic: {topic}")
            print(f"Depth: {depth}")
            print(f"Format: {format}\n")

            # Use the research template
            template_name = "research"
            template_version = "1.0.0"

            template_params = {
                "depth": depth,
                "format": format,
                "context": ""  # Add any additional context if needed
            }

            if stream:
                print("Findings:")
                async for chunk in self.agent.process_stream(
                    topic,
                    template=template_name,
                    template_version=template_version,
                    **template_params
                ):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = await self.agent.process(
                    topic,
                    template_name=template_name,
                    template_version=template_version,
                    **template_params
                )
                print(f"Findings:\n{response}\n")

        except Exception as e:
            print(f"Error conducting research: {e}")
            print(f"Make sure the template exists at: {self.research_prompts_dir}/research-1.0.0.yaml")


async def main():
    """Run the research agent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Research Agent using NazareAI Framework")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--depth", choices=["basic", "detailed", "comprehensive"],
                      default="detailed", help="Research depth")
    parser.add_argument("--format", choices=["markdown", "academic", "report"],
                      default="markdown", help="Output format")
    parser.add_argument("--no-stream", action="store_true",
                      help="Disable streaming output")
    args = parser.parse_args()

    # Create and run the research agent
    agent = ResearchAgent()
    
    try:
        await agent.initialize()
        await agent.research(
            args.topic,
            depth=args.depth,
            format=args.format,
            stream=not args.no_stream
        )
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    # Run the research agent
    asyncio.run(main()) 