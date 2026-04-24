"""
Redis connection management for LangGraph Q&A CLI Application
"""

import redis
from config import Config


class RedisConnection:
    def __init__(self):
        """Initialize Redis connection"""
        self.redis_client = None
        self.connect()
    
    def connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print("Connected to Redis successfully")
            return True
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            print("Make sure Redis is running: docker-compose up -d")
            self.redis_client = None
            return False
    
    def is_connected(self):
        """Check if Redis is connected"""
        return self.redis_client is not None
    
    def get_client(self):
        """Get Redis client instance"""
        return self.redis_client
    
    def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None
