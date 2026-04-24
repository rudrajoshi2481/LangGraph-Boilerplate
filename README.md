# LangGraph Q&A API

A basic Q&A API built with LangGraph and Ollama integration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Pull a model (if not already available):
```bash
ollama pull llama2
# or any other model like: ollama pull mistral, ollama pull codellama
```

## Running the API

```bash
cd part01
python app.py
```

The API will start on `http://localhost:8000`

## API Endpoints

### POST /ask
Ask a question and get an answer.

**Request:**
```json
{
    "question": "What is artificial intelligence?",
    "model_name": "llama2"  // optional, defaults to "llama2"
}
```

**Response:**
```json
{
    "answer": "Artificial intelligence is...",
    "model_used": "llama2",
    "processing_time": 2.34
}
```

### GET /
Root endpoint with API information.

### GET /health
Health check endpoint to verify Ollama connection.

## Usage Examples

### Using curl:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Python?", "model_name": "llama2"}'
```

### Using Python requests:
```python
import requests

response = requests.post("http://localhost:8000/ask", json={
    "question": "What is machine learning?",
    "model_name": "llama2"
})

print(response.json())
```

## Features

- **LangGraph Integration**: Uses LangGraph for structured question processing
- **Ollama Models**: Supports any model available in your Ollama installation
- **Async Processing**: FastAPI with async support for better performance
- **Error Handling**: Comprehensive error handling and logging
- **Health Checks**: Built-in health check endpoint
- **Flexible Models**: Easily switch between different Ollama models

## Available Models

You can use any model you have installed in Ollama. Common options:
- llama2
- mistral
- codellama
- vicuna
- and more...

To see available models: `ollama list`


AGENT_DEBUG=1 python ./app.py