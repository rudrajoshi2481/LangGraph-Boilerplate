"""
Conversation memory using Redis. Stores the last N (question, answer) turns
and renders them as context the model can see on each turn.
"""

import json
from typing import List, Dict
from redis_connection import RedisConnection

MEMORY_KEY = "conversation_memory"
DEFAULT_WINDOW = 20  # how many past turns to keep in context


class ConversationMemory:
    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self.redis_conn = RedisConnection()
    
    def add_turn(self, question: str, answer: str) -> None:
        if not self.redis_conn.is_connected():
            return
        client = self.redis_conn.get_client()
        client.rpush(MEMORY_KEY, json.dumps({"q": question, "a": answer}))
        # keep only the last `window` turns
        client.ltrim(MEMORY_KEY, -self.window, -1)
    
    def get_turns(self) -> List[Dict[str, str]]:
        if not self.redis_conn.is_connected():
            return []
        client = self.redis_conn.get_client()
        items = client.lrange(MEMORY_KEY, 0, -1)
        return [json.loads(x) for x in items]
    
    def render(self) -> str:
        """Render memory as a readable context block for the prompt."""
        turns = self.get_turns()
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"User: {t['q']}")
            lines.append(f"Assistant: {t['a']}")
        return "\n".join(lines)
    
    def clear(self) -> bool:
        if not self.redis_conn.is_connected():
            return False
        self.redis_conn.get_client().delete(MEMORY_KEY)
        return True
    
    def size(self) -> int:
        if not self.redis_conn.is_connected():
            return 0
        return self.redis_conn.get_client().llen(MEMORY_KEY)
