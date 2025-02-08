"""
Web interface for the NazareAI Framework.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nazare.core.agent import Agent, AgentConfig
from nazare.core.prompt import PromptLibrary
from nazare.models.openrouter import OpenRouterConfig, OpenRouterModel
from nazare.utils.config import load_settings


# Load settings from .env file
settings = load_settings()


# API models
class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str
    template_name: Optional[str] = None
    template_version: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None
    stream: bool = False


class TemplateInfo(BaseModel):
    """Response model for template information."""
    name: str
    version: str
    description: str
    author: str
    tags: List[str]


# Create FastAPI app
app = FastAPI(
    title="NazareAI Framework",
    description="Web interface for testing prompts and agents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global objects
agent: Optional[Agent] = None
prompt_library: Optional[PromptLibrary] = None


@app.on_event("startup")
async def startup_event():
    """Initialize global objects on startup."""
    global agent, prompt_library

    # Initialize prompt library
    prompt_library = PromptLibrary("prompts")

    # Create agent
    model_config = OpenRouterConfig(
        model_name=settings.default_model,
        api_key=settings.openrouter_api_key,
        api_base=settings.openrouter_api_base,
        timeout=settings.timeout,
        max_retries=settings.max_retries
    )
    model = OpenRouterModel(model_config)

    agent_config = AgentConfig(
        name="web-assistant",
        description="Web interface assistant",
        model_config=model_config,
        prompt_library_path="prompts",
        debug=settings.debug,
        timeout=settings.timeout,
        max_retries=settings.max_retries
    )

    agent = Agent(config=agent_config, model=model)
    await agent.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup global objects on shutdown."""
    global agent
    if agent:
        await agent.cleanup()


@app.get("/api/templates", response_model=List[TemplateInfo])
async def list_templates():
    """List available prompt templates."""
    if not prompt_library:
        raise HTTPException(status_code=503, detail="Prompt library not initialized")
    return prompt_library.list_templates()


@app.get("/api/templates/{name}")
async def get_template(name: str, version: Optional[str] = None):
    """Get a specific prompt template."""
    if not prompt_library:
        raise HTTPException(status_code=503, detail="Prompt library not initialized")
    try:
        template = prompt_library.get_template(name, version)
        return template.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/query")
async def process_query(request: QueryRequest):
    """Process a query through the agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        if request.stream:
            return StreamingResponse(
                agent.process_stream(
                    request.query,
                    template_name=request.template_name,
                    template_version=request.template_version,
                    **(request.template_params or {})
                ),
                media_type="text/event-stream"
            )
        else:
            response = await agent.process(
                request.query,
                template_name=request.template_name,
                template_version=request.template_version,
                **(request.template_params or {})
            )
            return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    if not agent:
        await websocket.close(code=1011, reason="Agent not initialized")
        return

    await websocket.accept()

    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue

            # Process query
            try:
                async for chunk in agent.process_stream(query):
                    await websocket.send_text(chunk)
                await websocket.send_json({"status": "complete"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NazareAI Framework</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            gap: 20px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .templates {
            flex: 1;
        }
        .query {
            flex: 2;
        }
        textarea {
            width: 100%;
            height: 150px;
            margin-bottom: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        select {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .response {
            white-space: pre-wrap;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            min-height: 200px;
            background-color: #fafafa;
        }
        #template-info {
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            margin-top: 10px;
        }
        h1, h2 {
            color: #333;
        }
    </style>
</head>
<body>
    <h1>NazareAI Framework</h1>
    <div class="container">
        <div class="templates">
            <h2>Templates</h2>
            <select id="template-select">
                <option value="">No template</option>
            </select>
            <div id="template-info"></div>
        </div>
        <div class="query">
            <h2>Query</h2>
            <textarea id="query-input" placeholder="Enter your query here..."></textarea>
            <button onclick="sendQuery()">Send</button>
            <div id="response" class="response"></div>
        </div>
    </div>

    <script>
        // Load templates on page load
        fetch('/api/templates')
            .then(response => response.json())
            .then(templates => {
                const select = document.getElementById('template-select');
                templates.forEach(template => {
                    const option = document.createElement('option');
                    option.value = template.name;
                    option.text = `${template.name} (${template.version})`;
                    select.appendChild(option);
                });
            });

        // Update template info when selected
        document.getElementById('template-select').addEventListener('change', function() {
            const name = this.value;
            if (name) {
                fetch(`/api/templates/${name}`)
                    .then(response => response.json())
                    .then(template => {
                        document.getElementById('template-info').innerHTML = `
                            <p><strong>Description:</strong> ${template.metadata.description}</p>
                            <p><strong>Author:</strong> ${template.metadata.author}</p>
                            <p><strong>Version:</strong> ${template.metadata.version}</p>
                        `;
                    });
            } else {
                document.getElementById('template-info').innerHTML = '';
            }
        });

        // Send query
        async function sendQuery() {
            const query = document.getElementById('query-input').value;
            const template = document.getElementById('template-select').value;
            const response = document.getElementById('response');
            
            response.innerHTML = 'Processing...';
            
            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        template_name: template || null,
                        stream: true
                    })
                });

                response.innerHTML = '';
                const reader = res.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    response.innerHTML += decoder.decode(value);
                }
            } catch (error) {
                response.innerHTML = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def get_index():
    """Serve the web interface."""
    return HTMLResponse(content=HTML_TEMPLATE)


def run():
    """Run the web interface."""
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port) 