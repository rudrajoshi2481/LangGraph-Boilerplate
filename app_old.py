import asyncio
import time
import json
from datetime import datetime
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
import redis

# LangGraph State
class QAState(TypedDict):
    question: str
    answer: str
    model_name: str

# Initialize Ollama LLM
def get_ollama_llm(model_name: str):
    """Initialize Ollama LLM with specified model"""
    try:
        return OllamaLLM(model=model_name, base_url="http://localhost:11434")
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        raise Exception(f"Failed to initialize Ollama model: {e}")

# LangGraph Node Functions
async def process_question(state: QAState) -> QAState:
    """Process the question using Ollama model"""
    try:
        llm = get_ollama_llm(state["model_name"])
        
        # Create a simple prompt
        prompt = f"Please answer the following question: {state['question']}"
        
        # Get response from the model
        response = await asyncio.to_thread(llm.invoke, prompt)
        
        state["answer"] = response
        print(f"Question processed successfully with model {state['model_name']}")
        return state
        
    except Exception as e:
        print(f"Error processing question: {e}")
        state["answer"] = f"Error processing question: {str(e)}"
        return state

def format_answer(state: QAState) -> QAState:
    """Format the final answer"""
    if not state["answer"].startswith("Error"):
        # Basic formatting
        state["answer"] = state["answer"].strip()
    return state

# Build the LangGraph
def build_qa_graph():
    """Build the Q&A LangGraph"""
    
    # Create the graph
    workflow = StateGraph(QAState)
    
    # Add nodes
    workflow.add_node("process_question", process_question)
    workflow.add_node("format_answer", format_answer)
    
    # Add edges
    workflow.set_entry_point("process_question")
    workflow.add_edge("process_question", "format_answer")
    workflow.add_edge("format_answer", END)
    
    # Compile the graph
    return workflow.compile()

# Create the compiled graph
qa_graph = build_qa_graph()

# Redis Chat Storage
class ChatStorage:
    def __init__(self, redis_host='localhost', redis_port=6380, redis_db=0):
        """Initialize Redis connection for chat storage"""
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print("Connected to Redis successfully")
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            print("Make sure Redis is running: docker-compose up -d")
            self.redis_client = None
    
    def save_chat(self, question: str, answer: str, model_name: str, processing_time: float):
        """Save a chat interaction to Redis"""
        if not self.redis_client:
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
            self.redis_client.set(chat_id, json.dumps(chat_data))
            
            # Add to chat history list (sorted by timestamp)
            self.redis_client.zadd('chat_history', {chat_id: float(datetime.now().timestamp())})
            
            # Keep only last 100 chats
            self.redis_client.zremrangebyrank('chat_history', 0, -101)
            
            print(f"Chat saved to Redis: {chat_id}")
            return True
            
        except Exception as e:
            print(f"Error saving chat to Redis: {e}")
            return False
    
    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """Get recent chat history from Redis"""
        if not self.redis_client:
            return []
        
        try:
            # Get recent chat IDs (sorted by timestamp, newest first)
            chat_ids = self.redis_client.zrevrange('chat_history', 0, limit - 1)
            
            chats = []
            for chat_id in chat_ids:
                chat_data = self.redis_client.get(chat_id)
                if chat_data:
                    chats.append(json.loads(chat_data))
            
            return chats
            
        except Exception as e:
            print(f"Error getting chat history from Redis: {e}")
            return []
    
    def clear_chat_history(self):
        """Clear all chat history from Redis"""
        if not self.redis_client:
            return False
        
        try:
            # Get all chat IDs
            chat_ids = self.redis_client.zrange('chat_history', 0, -1)
            
            # Delete each chat
            for chat_id in chat_ids:
                self.redis_client.delete(chat_id)
            
            # Clear the history list
            self.redis_client.delete('chat_history')
            
            print("Chat history cleared from Redis")
            return True
            
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get chat statistics from Redis"""
        if not self.redis_client:
            return {}
        
        try:
            total_chats = self.redis_client.zcard('chat_history')
            return {
                'total_chats': total_chats,
                'redis_connected': True
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'redis_connected': False}

# Initialize chat storage
chat_storage = ChatStorage()

def visualize_graph():
    """Visualize the LangGraph workflow"""
    try:
        # Print the graph structure
        print("=== LangGraph Structure ===")
        print(qa_graph.get_graph().draw_ascii())
        print("\n=== Graph Details ===")
        print(f"Nodes: {list(qa_graph.get_graph().nodes)}")
        print(f"Edges: {list(qa_graph.get_graph().edges)}")
        print(f"Entry point: {qa_graph.get_graph().nodes.get('__start__')}")
        
        # Try to create a visual representation
        try:
            # Try to save as PNG using different methods
            png_data = qa_graph.get_graph().draw_mermaid_png()
            with open("qa_graph.png", "wb") as f:
                f.write(png_data)
            print("\nGraph visualization saved as 'qa_graph.png'")
        except Exception as e:
            print(f"\nCould not create PNG image with mermaid: {e}")
            # Try alternative method
            try:
                # Try using graphviz directly
                from graphviz import Digraph
                dot = Digraph(comment='LangGraph Q&A Workflow')
                
                # Add nodes
                dot.node('start', '__start__', shape='circle')
                dot.node('process', 'process_question', shape='box')
                dot.node('format', 'format_answer', shape='box')
                dot.node('end', '__end__', shape='circle')
                
                # Add edges
                dot.edge('start', 'process')
                dot.edge('process', 'format')
                dot.edge('format', 'end')
                
                # Save as PNG
                dot.render('qa_graph_graphviz', format='png', cleanup=True)
                print("Graph visualization saved as 'qa_graph_graphviz.png' using Graphviz")
            except Exception as e2:
                print(f"Could not create PNG with Graphviz: {e2}")
                print("You may need to install Graphviz system package:")
                print("  Ubuntu/Debian: sudo apt-get install graphviz")
                print("  macOS: brew install graphviz")
                print("  Or visit: https://graphviz.org/download/")
        
        # Try to create Mermaid diagram
        try:
            mermaid_code = qa_graph.get_graph().draw_mermaid()
            print("\n=== Mermaid Diagram ===")
            print(mermaid_code)
            print("\nYou can paste this code into https://mermaid.live to see the diagram")
        except Exception as e:
            print(f"Could not generate Mermaid diagram: {e}")
            
    except Exception as e:
        print(f"Error visualizing graph: {e}")

async def ask_question(question: str, model_name: str = "qwen3.5:0.8b"):
    """Ask a question and get an answer using LangGraph"""
    
    start_time = time.time()
    
    try:
        # Create initial state
        initial_state = QAState(
            question=question,
            model_name=model_name
        )
        
        # Run the LangGraph
        result = await qa_graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        
        print(f"\nQuestion: {question}")
        print(f"Model: {model_name}")
        print(f"Processing Time: {round(processing_time, 2)}s")
        
        answer = result.get('answer', 'No answer received')
        if not answer or answer.strip() == '':
            answer = "Empty response received from model"
        
        print(f"\nAnswer: {answer}")
        print("-" * 50)
        
        # Save chat to Redis
        chat_storage.save_chat(question, answer, model_name, processing_time)
        
        return answer
        
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return None

def show_chat_history(limit: int = 10):
    """Display chat history"""
    chats = chat_storage.get_chat_history(limit)
    
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

def show_stats():
    """Display chat statistics"""
    stats = chat_storage.get_stats()
    
    print("\n=== Chat Statistics ===")
    if stats.get('redis_connected'):
        print(f"Total chats saved: {stats.get('total_chats', 0)}")
        print("Redis connection: Connected")
    else:
        print("Redis connection: Disconnected")
        print("Make sure Redis is running: docker-compose up -d")
    print("-" * 30)

def main():
    """Main CLI function"""
    print("=== LangGraph Q&A CLI ===")
    print("Commands:")
    print("  - Type your question to ask")
    print("  - 'graph' or 'visualize' - Show graph structure")
    print("  - 'history' or 'h' - Show chat history")
    print("  - 'stats' or 's' - Show chat statistics")
    print("  - 'clear' - Clear chat history")
    print("  - 'quit' or 'exit' - Exit the program")
    print("Default model: qwen3.5:0.8b")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question or command: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Check for graph visualization command
            if user_input.lower() in ['graph', 'visualize', 'show graph']:
                visualize_graph()
                continue
            
            # Check for chat history command
            if user_input.lower() in ['history', 'h', 'chat history']:
                show_chat_history()
                continue
            
            # Check for stats command
            if user_input.lower() in ['stats', 's', 'statistics']:
                show_stats()
                continue
            
            # Check for clear history command
            if user_input.lower() in ['clear', 'clear history', 'delete']:
                confirm = input("Are you sure you want to clear all chat history? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    chat_storage.clear_chat_history()
                else:
                    print("Operation cancelled.")
                continue
            
            # Check for empty input
            if not user_input:
                print("Please enter a question or command.")
                continue
            
            # Run the async function
            asyncio.run(ask_question(user_input))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()