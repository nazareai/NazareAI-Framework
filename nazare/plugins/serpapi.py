"""
SERP API plugin implementation for the NazareAI Framework.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
import requests

from nazare.core.plugin import Plugin, PluginMetadata


class SerpApiConfig(BaseModel):
    """Configuration for SERP API plugin."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    api_key: str = Field(..., description="SERP API key")
    engine: str = Field("google", description="Search engine to use")
    country: str = Field("us", description="Country code for search results")
    language: str = Field("en", description="Language code for search results")
    max_results: int = Field(10, description="Maximum number of results to return")
    timeout: float = Field(30.0, description="Timeout for API calls in seconds")


class SerpApiPlugin(Plugin):
    """Plugin for web search using SERP API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._base_url = "https://serpapi.com/search"

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="serpapi",
            version="0.1.0",
            description="Web search capabilities using SERP API",
            author="0xroyce",
            config_schema=SerpApiConfig
        )

    async def process(self, input_data: Any) -> Any:
        """
        Process a search query using SERP API.

        Args:
            input_data: Search query string or dictionary with search parameters

        Returns:
            Dictionary containing search results
        """
        try:
            # Convert input to search parameters
            if isinstance(input_data, str):
                search_params = {"q": input_data}
            elif isinstance(input_data, dict):
                search_params = input_data
            else:
                raise ValueError("Input must be a string or dictionary")

            # Build request parameters
            params = {
                "api_key": self.config["api_key"],
                "engine": self.config["engine"],
                "gl": self.config["country"],
                "hl": self.config["language"],
                "num": self.config["max_results"],
                **search_params
            }

            # Make API request
            response = requests.get(
                self._base_url,
                params=params,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            
            # Process and format results
            data = response.json()
            return self._format_results(data)

        except Exception as e:
            raise RuntimeError(f"Error processing search query: {e}")

    def _format_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw API response into structured results.

        Args:
            data: Raw API response data

        Returns:
            Formatted search results
        """
        formatted = {
            "query": data.get("search_parameters", {}).get("q", ""),
            "total_results": data.get("search_information", {}).get("total_results", 0),
            "organic_results": [],
            "knowledge_graph": None,
            "answer_box": None,
            "related_questions": []
        }

        # Process organic results
        if "organic_results" in data:
            formatted["organic_results"] = [
                {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0)
                }
                for result in data["organic_results"][:self.config["max_results"]]
            ]

        # Process knowledge graph if present
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            formatted["knowledge_graph"] = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", ""),
                "attributes": kg.get("attributes", {})
            }

        # Process answer box if present
        if "answer_box" in data:
            ab = data["answer_box"]
            formatted["answer_box"] = {
                "type": ab.get("type", ""),
                "answer": ab.get("answer", ""),
                "snippet": ab.get("snippet", ""),
                "source": ab.get("source", {})
            }

        # Process related questions if present
        if "related_questions" in data:
            formatted["related_questions"] = [
                {
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "source": q.get("source", {})
                }
                for q in data["related_questions"]
            ]

        return formatted

    async def pre_process(self, input_data: Any) -> Any:
        """
        Pre-process input before search.

        Args:
            input_data: Input data to pre-process

        Returns:
            Pre-processed input data
        """
        # Extract search queries from text if needed
        if isinstance(input_data, str):
            # Simple heuristic to extract search queries
            if "search for" in input_data.lower():
                query = input_data.lower().split("search for")[-1].strip()
                return {"q": query}
            return {"q": input_data}
        return input_data

    async def post_process(self, output_data: Any) -> Any:
        """
        Post-process search results.

        Args:
            output_data: Output data to post-process

        Returns:
            Post-processed output data
        """
        # If output_data is a string (e.g. from model response), return it as is
        if isinstance(output_data, str):
            return output_data
        
        # Handle dictionary output (e.g. from search results)
        if not output_data.get("organic_results"):
            return {"error": "No results found"}
        
        # Format results as markdown for better readability
        markdown = []
        
        # Add answer box if present
        if output_data.get("answer_box"):
            ab = output_data["answer_box"]
            markdown.append("### Quick Answer")
            markdown.append(ab["answer"])
            if ab.get("snippet"):
                markdown.append(f"\n{ab['snippet']}")
            markdown.append("")

        # Add knowledge graph if present
        if output_data.get("knowledge_graph"):
            kg = output_data["knowledge_graph"]
            markdown.append(f"### {kg['title']}")
            if kg.get("description"):
                markdown.append(kg["description"])
            if kg.get("attributes"):
                markdown.append("\n**Key Facts:**")
                for key, value in kg["attributes"].items():
                    markdown.append(f"- {key}: {value}")
            markdown.append("")

        # Add organic results
        markdown.append("### Search Results")
        for result in output_data["organic_results"]:
            markdown.append(f"**{result['title']}**")
            markdown.append(f"{result['snippet']}")
            markdown.append(f"[Read more]({result['link']})")
            markdown.append("")

        # Add related questions if present
        if output_data.get("related_questions"):
            markdown.append("### Related Questions")
            for question in output_data["related_questions"]:
                markdown.append(f"**Q: {question['question']}**")
                markdown.append(f"A: {question['answer']}")
                markdown.append("")

        return "\n".join(markdown) 