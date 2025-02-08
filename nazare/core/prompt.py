"""
Prompt management system for the NazareAI Framework.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
import yaml
from loguru import logger


class PromptMetadata(BaseModel):
    """Metadata for prompt templates."""
    name: str = Field(..., description="Name of the prompt template")
    version: str = Field(..., description="Version of the prompt template")
    description: str = Field("", description="Description of the prompt template")
    author: str = Field("", description="Author of the prompt template")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the prompt")
    parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameter definitions with their types and descriptions"
    )


class PromptTemplate:
    """A template for generating prompts with variable substitution."""

    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        default_values: Optional[Dict[str, Any]] = None
    ):
        self.template = template
        self.metadata = metadata
        self.default_values = default_values or {}

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with the provided values.

        Args:
            **kwargs: Values to substitute in the template

        Returns:
            Formatted prompt string
        """
        # Combine default values with provided values
        values = {**self.default_values, **kwargs}

        # Add empty string for optional parameters if not provided
        for param_name, param_info in self.metadata.parameters.items():
            if param_name not in values and param_info.get("required") is False:
                values[param_name] = ""

        # Validate all required parameters are provided
        required_params = {
            name for name, param in self.metadata.parameters.items()
            if not param.get("required") is False  # Parameters are required by default
        }
        missing_params = required_params - set(values.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        return self.template.format(**values)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary representation.

        Returns:
            Dictionary containing template data
        """
        return {
            "template": self.template,
            "metadata": self.metadata.dict(),
            "default_values": self.default_values
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """
        Create a template from a dictionary representation.

        Args:
            data: Dictionary containing template data

        Returns:
            PromptTemplate instance
        """
        # Handle both root-level metadata and nested metadata
        if "metadata" in data:
            # Template is in the expected format
            return cls(
                template=data["template"],
                metadata=PromptMetadata(**data["metadata"]),
                default_values=data.get("default_values", {})
            )
        else:
            # Template has metadata at root level
            metadata_fields = {
                "name", "version", "description", "author", "tags", "parameters"
            }
            metadata = {
                key: data[key] for key in metadata_fields
                if key in data
            }
            return cls(
                template=data["template"],
                metadata=PromptMetadata(**metadata),
                default_values=data.get("default_values", {})
            )


class PromptLibrary:
    """Manager for prompt templates with versioning support."""

    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        logger.debug(f"Initializing PromptLibrary with storage path: {self.storage_path}")
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all templates from storage."""
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)
            logger.debug("Created prompts directory as it did not exist")
            return

        logger.debug(f"Loading templates from {self.storage_path}")
        
        # List all files in the directory for debugging
        logger.debug("Listing all files in prompts directory:")
        for path in self.storage_path.rglob("*"):
            logger.debug(f"Found file: {path}")

        # Try loading each YAML/JSON file
        yaml_files = list(self.storage_path.rglob("*.yaml"))
        yml_files = list(self.storage_path.rglob("*.yml"))
        json_files = list(self.storage_path.rglob("*.json"))
        
        logger.debug(f"Found YAML files: {yaml_files}")
        logger.debug(f"Found YML files: {yml_files}")
        logger.debug(f"Found JSON files: {json_files}")

        for file_path in yaml_files + yml_files + json_files:
            try:
                logger.debug(f"Loading template from {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.suffix == ".json":
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                logger.debug(f"Loaded template data: {data}")

                # Handle both root-level metadata and nested metadata
                if "metadata" in data:
                    logger.debug("Found nested metadata structure")
                else:
                    logger.debug("Found root-level metadata structure")
                    metadata_fields = {
                        "name", "version", "description", "author", "tags", "parameters"
                    }
                    logger.debug(f"Available metadata fields: {[f for f in metadata_fields if f in data]}")

                template = PromptTemplate.from_dict(data)
                logger.debug(f"Created template object: {template.metadata.name} v{template.metadata.version}")
                self.add_template(template, save=False)
                logger.debug(f"Added template to library: {template.metadata.name} v{template.metadata.version}")
                logger.debug(f"Current templates: {self.templates}")
            except Exception as e:
                logger.error(f"Error loading template from {file_path}: {e}", exc_info=True)

    def add_template(
        self,
        template: PromptTemplate,
        save: bool = True
    ) -> None:
        """
        Add a template to the library.

        Args:
            template: Template to add
            save: Whether to save the template to storage
        """
        name = template.metadata.name
        version = template.metadata.version

        if name not in self.templates:
            self.templates[name] = {}

        self.templates[name][version] = template

        if save:
            self._save_template(template)

    def get_template(
        self,
        name: str,
        version: Optional[str] = None
    ) -> PromptTemplate:
        """
        Get a template by name and optionally version.

        Args:
            name: Name of the template
            version: Optional version of the template

        Returns:
            PromptTemplate instance
        """
        logger.debug(f"Getting template: {name} v{version or 'latest'}")
        logger.debug(f"Available templates: {self.templates}")
        
        if name not in self.templates:
            raise ValueError(f"Template {name} not found")

        if version is None:
            # Get latest version
            version = max(self.templates[name].keys())
            logger.debug(f"Using latest version: {version}")

        if version not in self.templates[name]:
            raise ValueError(f"Version {version} of template {name} not found")

        return self.templates[name][version]

    def _save_template(self, template: PromptTemplate) -> None:
        """
        Save a template to storage.

        Args:
            template: Template to save
        """
        name = template.metadata.name
        version = template.metadata.version
        filename = f"{name}-{version}.yaml"

        # Create subdirectory based on template name
        template_dir = self.storage_path / name
        template_dir.mkdir(parents=True, exist_ok=True)

        file_path = template_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False)

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.

        Returns:
            List of template information dictionaries
        """
        templates = []
        for name, versions in self.templates.items():
            for version, template in versions.items():
                templates.append({
                    "name": name,
                    "version": version,
                    "description": template.metadata.description,
                    "author": template.metadata.author,
                    "tags": template.metadata.tags
                })
        return templates

    def search_templates(
        self,
        query: str,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search templates by name, description, or tags.

        Args:
            query: Search query string
            tags: Optional list of tags to filter by

        Returns:
            List of matching template information dictionaries
        """
        query = query.lower()
        results = []

        for template_info in self.list_templates():
            # Check if template matches query
            if (query in template_info["name"].lower() or
                query in template_info["description"].lower()):
                
                # Check if template has all required tags
                if tags:
                    if not all(tag in template_info["tags"] for tag in tags):
                        continue
                
                results.append(template_info)

        return results 