r"""
LangGraph construction and visualization for Q&A CLI Application.

The graph explicitly models the agent flow:

    __start__
        v
    build_prompt
        v
    agent_think  <-----------+
        v                    |
    route_tool               |
     / | | \                 |
bash pdf fact finalize ------+ (each tool loops back)
                 v
             __end__
"""

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from config import Config
from ollama_client import OllamaClient
from memory import ConversationMemory
from prompts import SYSTEM_PROMPT, TOOL_PROMPT
from agent import ToolAgent, TOOL_PATTERN, _looks_like_tool_question, DEBUG
from tools import bash_tool, pdf_tool, fact_check_tool

MAX_TOOL_STEPS = 4


class QAState(TypedDict, total=False):
    question: str
    answer: str
    model_name: str
    # Agent-loop fields
    prompt: str        # rolling prompt fed to the model
    think_output: str  # latest model output
    tool_name: str     # parsed tool name, empty if none
    tool_args: str     # parsed tool arguments
    steps: int         # tool-call counter
    nudged: bool       # whether we've already nudged for a missing tool
    last_call: str     # "<name>|<args>" of previous tool call


class GraphBuilder:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.memory = ConversationMemory()
        self.agent = ToolAgent(self.ollama_client)

    # ------------------------------------------------------------------
    # Node: build_prompt
    # ------------------------------------------------------------------
    def build_prompt_node(self, state: QAState) -> QAState:
        parts = [SYSTEM_PROMPT.strip()]
        mem = self.memory.render()
        if mem:
            parts.append("Previous conversation:\n" + mem)
        parts.append(TOOL_PROMPT.strip())
        parts.append(f"User: {state['question']}\nAssistant:")
        prompt = "\n\n".join(parts)
        
        new_state = state.copy()
        new_state["prompt"] = prompt
        new_state["steps"] = 0
        new_state["nudged"] = False
        new_state["last_call"] = ""
        return new_state

    # ------------------------------------------------------------------
    # Node: agent_think - ask the model what to do next
    # ------------------------------------------------------------------
    async def agent_think_node(self, state: QAState) -> QAState:
        out = await self.agent._generate_async(state["prompt"], state["model_name"])
        
        new_state = state.copy()
        new_state["think_output"] = out
        
        match = TOOL_PATTERN.search(out)
        if match:
            new_state["tool_name"] = match.group(1).lower()
            new_state["tool_args"] = match.group(2).strip()
        else:
            new_state["tool_name"] = ""
            new_state["tool_args"] = ""
        return new_state

    # ------------------------------------------------------------------
    # Conditional router: pick next node based on think_output
    # ------------------------------------------------------------------
    def route_tool(self, state: QAState) -> str:
        tool = state.get("tool_name", "")
        steps = state.get("steps", 0)
        
        # Stop if we hit the step limit.
        if steps >= MAX_TOOL_STEPS:
            return "finalize"
        
        # Model emitted a TOOL: line -> route to that tool.
        if tool == "bash":
            return "bash_tool"
        if tool == "pdf":
            return "pdf_tool"
        if tool == "fact":
            return "fact_tool"
        
        # No tool call. If the user's question clearly needs one and we
        # haven't nudged yet, ask the model to try again.
        if (
            not state.get("nudged")
            and _looks_like_tool_question(state.get("question", ""))
            and not state.get("last_call")
        ):
            return "nudge"
        
        # Otherwise we're done thinking - produce the final answer.
        return "finalize"

    # ------------------------------------------------------------------
    # Node: nudge - reprompt model to use a tool
    # ------------------------------------------------------------------
    def nudge_node(self, state: QAState) -> QAState:
        new_state = state.copy()
        new_state["prompt"] = (
            state["prompt"]
            + "\n\nReminder: the user asked about the real system. You MUST "
            "call a tool. Output a single TOOL: line now and nothing else.\n"
            "Assistant:"
        )
        new_state["nudged"] = True
        return new_state

    # ------------------------------------------------------------------
    # Helper: run a tool, check for repeats, update rolling prompt
    # ------------------------------------------------------------------
    def _run_and_accumulate(self, state: QAState, result: str) -> QAState:
        tool_name = state["tool_name"]
        tool_args = state["tool_args"]
        call_key = f"{tool_name}|{tool_args}"
        
        new_state = state.copy()
        new_state["steps"] = state.get("steps", 0) + 1
        
        # Repeat-call guard: if same call as last time, short-circuit by
        # treating this as the final answer (handled by finalize_node).
        if call_key == state.get("last_call", "") and state.get("last_call"):
            new_state["answer"] = result
            new_state["last_call"] = call_key
            new_state["steps"] = MAX_TOOL_STEPS  # force finalize
            return new_state
        
        new_state["last_call"] = call_key
        new_state["answer"] = result  # latest observation as fallback answer
        
        # Append observation + finalize instruction for the next think step.
        new_state["prompt"] = (
            state["prompt"]
            + "\n" + state["think_output"]
            + f"\nOBSERVATION: {result}\n"
            + "Now write the final answer to the user, citing the actual "
            "values from the OBSERVATION above. If you need more info, you "
            "may call another TOOL.\nAssistant:"
        )
        if DEBUG:
            truncated = result if len(result) < 200 else result[:200] + "..."
            print(f"[graph] {tool_name} -> {truncated}")
        return new_state

    # ------------------------------------------------------------------
    # Tool nodes (one per tool - visible in the graph)
    # ------------------------------------------------------------------
    def bash_tool_node(self, state: QAState) -> QAState:
        result = bash_tool(state["tool_args"])
        return self._run_and_accumulate(state, result)

    def pdf_tool_node(self, state: QAState) -> QAState:
        result = pdf_tool(state["tool_args"])
        return self._run_and_accumulate(state, result)

    def fact_tool_node(self, state: QAState) -> QAState:
        result = fact_check_tool(
            state["tool_args"], self.ollama_client, state["model_name"]
        )
        return self._run_and_accumulate(state, result)

    # ------------------------------------------------------------------
    # Node: finalize - produce the final user-facing answer
    # ------------------------------------------------------------------
    def finalize_node(self, state: QAState) -> QAState:
        new_state = state.copy()
        # If the last think step produced a plain answer (no TOOL), use it.
        if not state.get("tool_name") and state.get("think_output"):
            new_state["answer"] = state["think_output"].strip()
        # Otherwise fall back to the last observation stored in answer.
        response = new_state.get("answer", "").strip() or "(no answer)"
        new_state["answer"] = response
        
        print(f"\nAnswer: {response}")
        
        # Save turn to memory.
        if response and not response.startswith("[error]"):
            self.memory.add_turn(state["question"], response)
        return new_state

    # ==================================================================
    # Assemble the graph
    # ==================================================================
    def build_qa_graph(self):
        workflow = StateGraph(QAState)
        
        workflow.add_node("build_prompt", self.build_prompt_node)
        workflow.add_node("agent_think", self.agent_think_node)
        workflow.add_node("nudge", self.nudge_node)
        workflow.add_node("bash_tool", self.bash_tool_node)
        workflow.add_node("pdf_tool", self.pdf_tool_node)
        workflow.add_node("fact_tool", self.fact_tool_node)
        workflow.add_node("finalize", self.finalize_node)
        
        workflow.set_entry_point("build_prompt")
        workflow.add_edge("build_prompt", "agent_think")
        
        # After thinking, conditionally route to one of the tools, to
        # nudge, or to finalize.
        workflow.add_conditional_edges(
            "agent_think",
            self.route_tool,
            {
                "bash_tool": "bash_tool",
                "pdf_tool": "pdf_tool",
                "fact_tool": "fact_tool",
                "nudge": "nudge",
                "finalize": "finalize",
            },
        )
        
        # Every tool and the nudge loop back to agent_think.
        workflow.add_edge("bash_tool", "agent_think")
        workflow.add_edge("pdf_tool", "agent_think")
        workflow.add_edge("fact_tool", "agent_think")
        workflow.add_edge("nudge", "agent_think")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def visualize_graph(self, qa_graph):
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
                with open(Config.GRAPH_PNG_FILENAME, "wb") as f:
                    f.write(png_data)
                print(f"\nGraph visualization saved as '{Config.GRAPH_PNG_FILENAME}'")
            except Exception as e:
                print(f"\nCould not create PNG image with mermaid: {e}")
                # Fallback: use Graphviz with the REAL nodes/edges from the
                # compiled LangGraph, not hard-coded ones.
                try:
                    from graphviz import Digraph
                    g = qa_graph.get_graph()
                    dot = Digraph(comment="LangGraph Q&A Workflow")
                    for node_id in g.nodes:
                        shape = "circle" if node_id in ("__start__", "__end__") else "box"
                        dot.node(node_id, node_id, shape=shape)
                    for edge in g.edges:
                        dot.edge(edge.source, edge.target)
                    dot.render(Config.GRAPH_GRAPHVIZ_FILENAME, format="png", cleanup=True)
                    print(
                        f"Graph visualization saved as "
                        f"'{Config.GRAPH_GRAPHVIZ_FILENAME}.png' using Graphviz"
                    )
                except Exception as e2:
                    print(f"Could not create PNG with Graphviz: {e2}")
                    print("You may need to install Graphviz system package:")
                    print("  Ubuntu/Debian: sudo apt-get install graphviz")
                    print("  macOS: brew install graphviz")
            
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
