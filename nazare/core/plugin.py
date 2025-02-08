"""
Plugin system implementation for the NazareAI Framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class PluginMetadata(BaseModel):
    """Metadata for plugins."""
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: str = Field("", description="Description of the plugin")
    author: str = Field("", description="Author of the plugin")
    dependencies: List[str] = Field(default_factory=list, description="List of plugin dependencies")
    config_schema: Optional[Type[BaseModel]] = Field(None, description="Configuration schema for the plugin")


class Plugin(ABC):
    """Base class for all plugins in the framework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
        self._initialized = False

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    def _validate_config(self) -> None:
        """Validate plugin configuration."""
        if self.metadata.config_schema and self.config:
            self.config = self.metadata.config_schema(**self.config)

    async def initialize(self) -> None:
        """Initialize the plugin."""
        if self._initialized:
            return
        await self._initialize()
        self._initialized = True

    async def _initialize(self) -> None:
        """Internal initialization logic."""
        pass

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if not self._initialized:
            return
        await self._cleanup()
        self._initialized = False

    async def _cleanup(self) -> None:
        """Internal cleanup logic."""
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input data according to plugin functionality.

        Args:
            input_data: Input data to process

        Returns:
            Processed output data
        """
        pass

    async def pre_process(self, input_data: Any) -> Any:
        """
        Pre-process input data before main processing.

        Args:
            input_data: Input data to pre-process

        Returns:
            Pre-processed input data
        """
        return input_data

    async def post_process(self, output_data: Any) -> Any:
        """
        Post-process output data after main processing.

        Args:
            output_data: Output data to post-process

        Returns:
            Post-processed output data
        """
        return output_data

    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


class PluginManager:
    """Manager for handling plugin lifecycle and dependencies."""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._initialized = False

    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin with the manager.

        Args:
            plugin: Plugin instance to register
        """
        if plugin.metadata.name in self._plugins:
            raise ValueError(f"Plugin {plugin.metadata.name} is already registered")
        
        # Check dependencies
        for dep in plugin.metadata.dependencies:
            if dep not in self._plugins:
                raise ValueError(f"Plugin {plugin.metadata.name} depends on {dep} which is not registered")
        
        self._plugins[plugin.metadata.name] = plugin

    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin from the manager.

        Args:
            plugin_name: Name of the plugin to unregister
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} is not registered")
        
        # Check if other plugins depend on this one
        for name, plugin in self._plugins.items():
            if plugin_name in plugin.metadata.dependencies:
                raise ValueError(f"Cannot unregister {plugin_name} as {name} depends on it")
        
        del self._plugins[plugin_name]

    def get_plugin(self, plugin_name: str) -> Plugin:
        """
        Get a plugin by name.

        Args:
            plugin_name: Name of the plugin to retrieve

        Returns:
            Plugin instance
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} is not registered")
        return self._plugins[plugin_name]

    async def initialize_all(self) -> None:
        """Initialize all registered plugins in dependency order."""
        if self._initialized:
            return

        # Create dependency graph and initialize in order
        initialized = set()
        while len(initialized) < len(self._plugins):
            for name, plugin in self._plugins.items():
                if name in initialized:
                    continue
                if all(dep in initialized for dep in plugin.metadata.dependencies):
                    await plugin.initialize()
                    initialized.add(name)

        self._initialized = True

    async def cleanup_all(self) -> None:
        """Cleanup all registered plugins in reverse dependency order."""
        if not self._initialized:
            return

        # Create reverse dependency graph and cleanup in order
        cleaned = set()
        while len(cleaned) < len(self._plugins):
            for name, plugin in self._plugins.items():
                if name in cleaned:
                    continue
                dependent_plugins = [
                    p for p in self._plugins.values()
                    if name in p.metadata.dependencies and p.metadata.name not in cleaned
                ]
                if not dependent_plugins:
                    await plugin.cleanup()
                    cleaned.add(name)

        self._initialized = False 