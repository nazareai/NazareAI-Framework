# NazareAI Framework

A powerful and flexible framework for building AI agents with modular architecture and extensive integration capabilities.

## Features

- **Advanced Model Integration**
  - OpenRouter integration with Claude-3 and other leading models
  - Streaming response support
  - Configurable model parameters and settings

- **Intelligent Research Capabilities**
  - Built-in research agent with web search integration
  - Configurable research depth and output formats
  - Academic and report-style formatting options

- **Modular Prompt System**
  - YAML-based prompt template management
  - Version control for templates
  - Dynamic parameter substitution
  - Context-aware prompt generation

- **Plugin Architecture**
  - Extensible plugin system
  - Pre/post-processing hooks
  - Built-in SerpAPI integration for web search
  - Easy custom plugin development

- **Robust Configuration**
  - Environment-based configuration
  - Type-safe settings with Pydantic
  - Flexible model and plugin configuration
  - Debug mode support

- **Developer Experience**
  - Async/await support throughout
  - Comprehensive logging system
  - Command-line interface tools
  - Clear error handling and debugging

- **Production Ready**
  - Exception handling and retries
  - Resource cleanup
  - Configurable timeouts
  - API key management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nazareai/NazareAI-Framework.git
   cd NazareAI-Framework
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

### Basic Usage

```python
from nazare.core import Agent, AgentConfig
from nazare.models import OpenRouterConfig, OpenRouterModel
from nazare.plugins import SerpApiConfig, SerpApiPlugin
from nazare.utils.config import load_settings

# Load settings from .env
settings = load_settings()

# Configure model
model_config = OpenRouterConfig(
    model_name="anthropic/claude-3-opus-20240229",
    api_key=settings.openrouter_api_key
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

# Process queries
response = await agent.process("What is the capital of France?")
print(response)

# Cleanup
await agent.cleanup()
```

### Research Agent Example

```python
# Create a research agent with search capabilities
model_config = OpenRouterConfig(
    model_name=settings.default_model,
    api_key=settings.openrouter_api_key
)
model = OpenRouterModel(model_config)

# Configure search plugin
serpapi_config = SerpApiConfig(
    api_key=settings.serpapi_api_key,
    max_results=10
)
search_plugin = SerpApiPlugin(config=serpapi_config.model_dump())

# Configure agent
agent_config = AgentConfig(
    name="research-assistant",
    description="An agent that conducts thorough research",
    model_settings=model_config,
    prompt_library_path="prompts"
)

# Create agent with plugin
agent = Agent(config=agent_config, model=model, plugins=[search_plugin])
await agent.initialize()

# Research with template
response = await agent.process(
    "What are the latest developments in fusion energy?",
    template_name="research",
    template_version="1.0.0",
    depth="comprehensive",
    format="academic"
)
print(response)

await agent.cleanup()
```

### Command Line Usage

The framework includes ready-to-use command line tools:

```bash
# Basic agent
python examples/basic_agent.py

# Research agent with options
python examples/research_agent.py "your research topic" --depth comprehensive --format academic
```

## Project Structure

```
nazare/
├── core/           # Core framework components
├── models/         # Model integrations
├── plugins/        # Plugin system
├── prompts/        # Prompt templates
├── security/       # Security features
├── utils/          # Utility functions
└── web/           # Web interface
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache 2.0 Licence.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
