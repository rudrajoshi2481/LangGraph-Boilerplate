"""
Ollama client management for LangGraph Q&A CLI Application
"""

import sys
import asyncio
import requests
from langchain_ollama import OllamaLLM
from config import Config


class OllamaClient:
    def __init__(self):
        """Initialize Ollama client"""
        self.base_url = Config.OLLAMA_BASE_URL
    
    def get_llm(self, model_name: str = Config.DEFAULT_MODEL):
        """Initialize Ollama LLM with specified model"""
        try:
            return OllamaLLM(model=model_name, base_url=self.base_url)
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")
            raise Exception(f"Failed to initialize Ollama model: {e}")
    
    async def stream_response(self, prompt: str, model_name: str) -> str:
        """Stream response from Ollama, printing each token as it arrives.
        Thinking is disabled so reasoning models (qwen3, deepseek-r1, etc.) answer directly."""
        import json
        
        print("\nAnswer: ", end="", flush=True)
        
        def _stream():
            full = ""
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "think": False,  # skip <think> reasoning for reasoning models
            }
            
            with requests.post(url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    
                    token = chunk.get("response", "")
                    if token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        full += token
                    
                    if chunk.get("done"):
                        break
            return full
        
        try:
            result = await asyncio.to_thread(_stream)
            print()  # trailing newline after the streamed answer
            return result.strip()
        except Exception as e:
            print(f"\nError streaming from Ollama: {e}")
            raise Exception(f"Failed to stream model response: {e}")
    
    def test_connection(self, model_name: str = Config.DEFAULT_MODEL) -> bool:
        """Test Ollama connection"""
        try:
            llm = self.get_llm(model_name)
            response = llm.invoke("Hello")
            return True
        except Exception as e:
            print(f"Ollama connection test failed: {e}")
            return False
