"""
Configuration settings for LangGraph Q&A CLI Application
"""

class Config:
    # Ollama Settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen3.5:latest"
    
    # Redis Settings
    REDIS_HOST = "localhost"
    REDIS_PORT = 6380
    REDIS_DB = 0
    
    # Chat Settings
    MAX_CHAT_HISTORY = 100
    
    # Graph Settings
    GRAPH_PNG_FILENAME = "qa_graph.png"
    GRAPH_GRAPHVIZ_FILENAME = "qa_graph_graphviz"
    
    # CLI Settings
    CLI_PROMPT = "Your question or command: "
    CLI_SEPARATOR = "-" * 50
