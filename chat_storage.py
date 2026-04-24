"""
Chat storage management for LangGraph Q&A CLI Application
"""

import json
from datetime import datetime
from typing import List, Dict
from config import Config
from redis_connection import RedisConnection


class ChatStorage:
    def __init__(self):
        """Initialize chat storage with Redis connection"""
        self.redis_conn = RedisConnection()
    
    def save_chat(self, question: str, answer: str, model_name: str, processing_time: float) -> bool:
        """Save a chat interaction to Redis"""
        if not self.redis_conn.is_connected():
            return False
        
        try:
            chat_data = {
                'question': question,
                'answer': answer,
                'model_name': model_name,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate unique ID for this chat
            chat_id = f"chat:{datetime.now().timestamp()}"
            
            # Save chat data
            redis_client = self.redis_conn.get_client()
            redis_client.set(chat_id, json.dumps(chat_data))
            
            # Add to chat history list (sorted by timestamp)
            redis_client.zadd('chat_history', {chat_id: float(datetime.now().timestamp())})
            
            # Keep only last MAX_CHAT_HISTORY chats
            redis_client.zremrangebyrank('chat_history', 0, -(Config.MAX_CHAT_HISTORY + 1))
            
            print(f"Chat saved to Redis: {chat_id}")
            return True
            
        except Exception as e:
            print(f"Error saving chat to Redis: {e}")
            return False
    
    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """Get recent chat history from Redis"""
        if not self.redis_conn.is_connected():
            return []
        
        try:
            # Get recent chat IDs (sorted by timestamp, newest first)
            redis_client = self.redis_conn.get_client()
            chat_ids = redis_client.zrevrange('chat_history', 0, limit - 1)
            
            chats = []
            for chat_id in chat_ids:
                chat_data = redis_client.get(chat_id)
                if chat_data:
                    chats.append(json.loads(chat_data))
            
            return chats
            
        except Exception as e:
            print(f"Error getting chat history from Redis: {e}")
            return []
    
    def clear_chat_history(self) -> bool:
        """Clear all chat history from Redis"""
        if not self.redis_conn.is_connected():
            return False
        
        try:
            # Get all chat IDs
            redis_client = self.redis_conn.get_client()
            chat_ids = redis_client.zrange('chat_history', 0, -1)
            
            # Delete each chat
            for chat_id in chat_ids:
                redis_client.delete(chat_id)
            
            # Clear the history list
            redis_client.delete('chat_history')
            
            print("Chat history cleared from Redis")
            return True
            
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get chat statistics from Redis"""
        if not self.redis_conn.is_connected():
            return {'redis_connected': False}
        
        try:
            redis_client = self.redis_conn.get_client()
            total_chats = redis_client.zcard('chat_history')
            return {
                'total_chats': total_chats,
                'redis_connected': True
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'redis_connected': False}
