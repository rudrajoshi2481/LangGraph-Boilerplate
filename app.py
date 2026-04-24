"""
Main CLI application for LangGraph Q&A with Redis storage
"""

import asyncio
import time
from typing import List, Dict
from config import Config
from chat_storage import ChatStorage
from graph_builder import GraphBuilder


class LangGraphQA:
    def __init__(self):
        """Initialize the LangGraph Q&A CLI application"""
        self.chat_storage = ChatStorage()
        self.graph_builder = GraphBuilder()
        self.qa_graph = self.graph_builder.build_qa_graph()
    
    async def ask_question(self, question: str, model_name: str = Config.DEFAULT_MODEL):
        """Ask a question and get an answer using LangGraph"""
        
        start_time = time.time()
        
        try:
            # Create initial state
            from graph_builder import QAState
            initial_state = QAState(
                question=question,
                model_name=model_name
            )
            
            # Run the LangGraph
            result = await self.qa_graph.ainvoke(initial_state)
            
            processing_time = time.time() - start_time
            
            answer = result.get('answer', '')
            if not answer or answer.strip() == '':
                answer = "Empty response received from model"
            
            print(f"\n[Model: {model_name} | Time: {round(processing_time, 2)}s]")
            print(Config.CLI_SEPARATOR)
            
            # Save chat to Redis
            self.chat_storage.save_chat(question, answer, model_name, processing_time)
            
            return answer
            
        except Exception as e:
            print(f"Error in ask_question: {e}")
            return None
    
    def show_chat_history(self, limit: int = 10):
        """Display chat history"""
        chats = self.chat_storage.get_chat_history(limit)
        
        if not chats:
            print("No chat history found.")
            return
        
        print(f"\n=== Recent Chat History (Last {len(chats)} chats) ===")
        for i, chat in enumerate(chats, 1):
            timestamp = chat.get('timestamp', 'Unknown time')
            print(f"\n[{i}] {timestamp}")
            print(f"Q: {chat.get('question', 'N/A')}")
            print(f"A: {chat.get('answer', 'N/A')[:100]}{'...' if len(chat.get('answer', '')) > 100 else ''}")
            print(f"Model: {chat.get('model_name', 'N/A')} | Time: {chat.get('processing_time', 'N/A')}s")
            print("-" * 30)
    
    def show_stats(self):
        """Display chat statistics"""
        stats = self.chat_storage.get_stats()
        
        print("\n=== Chat Statistics ===")
        if stats.get('redis_connected'):
            print(f"Total chats saved: {stats.get('total_chats', 0)}")
            print("Redis connection: Connected")
        else:
            print("Redis connection: Disconnected")
            print("Make sure Redis is running: docker-compose up -d")
        print("-" * 30)
    
    def visualize_graph(self):
        """Visualize the LangGraph workflow"""
        self.graph_builder.visualize_graph(self.qa_graph)
    
    def print_help(self):
        print("""
Commands:
  <anything>      ask the assistant - it will use tools on its own if needed
  history         show recent chats
  stats           show stats
  graph           show LangGraph structure (+ save PNG)
  clear history   wipe chat history
  clear memory    wipe conversation memory
  clear all       wipe both history and memory
  help            show this help
  quit            exit
""")
    
    def handle_command(self, user_input: str) -> bool:
        """Handle CLI commands. Returns True if handled, False otherwise."""
        low = user_input.lower().strip()
        
        if low in ['quit', 'exit', 'q']:
            raise SystemExit
        if low in ['help', '?']:
            self.print_help()
            return True
        if low in ['graph', 'visualize']:
            self.visualize_graph()
            return True
        if low in ['history', 'h']:
            self.show_chat_history()
            return True
        if low in ['stats', 's']:
            self.show_stats()
            return True
        if low == 'clear history':
            self.chat_storage.clear_chat_history()
            return True
        if low == 'clear memory':
            self.graph_builder.memory.clear()
            print("Conversation memory cleared.")
            return True
        if low in ['clear all', 'clear']:
            self.chat_storage.clear_chat_history()
            self.graph_builder.memory.clear()
            print("Chat history AND memory cleared.")
            return True
        
        return False
    
    def run_cli(self):
        """Run the main CLI interface"""
        print("=== LangGraph Q&A CLI ===")
        print(f"Model: {Config.DEFAULT_MODEL}  |  type 'help' for commands")
        print(Config.CLI_SEPARATOR)
        
        while True:
            try:
                user_input = input(f"\n{Config.CLI_PROMPT}").strip()
                if not user_input:
                    continue
                
                if self.handle_command(user_input):
                    continue
                
                asyncio.run(self.ask_question(user_input))
                
            except (KeyboardInterrupt, SystemExit):
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    app = LangGraphQA()
    app.run_cli()


if __name__ == "__main__":
    main()
