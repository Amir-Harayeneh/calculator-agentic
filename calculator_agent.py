"""
LangGraph Calculator Agent for LangGraph Studio
Simple calculator agent with bind_tools pattern
"""

import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import operator
from pathlib import Path

# Try to import dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# ============================================================================
# 📌 API KEYS CONFIGURATION FROM .ENV FILE
# ============================================================================

def load_api_keys_from_env(env_path: str = None):
    """
    Load API keys from .env file

    Args:
        env_path: Path to .env file. If None, looks for .env in current directory.

    Your .env file should contain:
        OPENAI_API_KEY=sk-...
    """

    if not DOTENV_AVAILABLE:
        print("⚠️  python-dotenv not installed. Using environment variables.")
        return True

    # Use provided path or look in current directory
    if env_path:
        env_file_path = env_path
    else:
        # Look for .env in current directory
        env_file_path = Path(".env")

    # Check if file exists
    if not Path(env_file_path).exists():
        print(f"⚠️  .env file not found. Using environment variables.")
        return True

    # Load .env file
    load_dotenv(env_file_path, override=True)
    print(f"✅ .env file loaded successfully!")

    # Check OpenAI key
    if os.getenv("OPENAI_API_KEY"):
        print(f"✅ OPENAI_API_KEY: Found")
    else:
        print(f"❌ OPENAI_API_KEY: Not found")
        return False

    return True


# ============================================================================
# 🛠️ CALCULATOR TOOLS
# ============================================================================

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Sum of a and b
    """
    result = a + b
    print(f"   🔧 Tool: add({a}, {b}) = {result}")
    return result


@tool
def subtract(a: float, b: float) -> float:
    """
    Subtract second number from first number.
    
    Args:
        a: First number (minuend)
        b: Second number (subtrahend)
    
    Returns:
        Difference of a and b
    """
    result = a - b
    print(f"   🔧 Tool: subtract({a}, {b}) = {result}")
    return result


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Product of a and b
    """
    result = a * b
    print(f"   🔧 Tool: multiply({a}, {b}) = {result}")
    return result


@tool
def divide(a: float, b: float) -> float:
    """
    Divide first number by second number.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        Quotient of a divided by b
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    
    result = a / b
    print(f"   🔧 Tool: divide({a}, {b}) = {result}")
    return result


# ============================================================================
# 📊 STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for calculator agent with message history"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================================
# 🤖 CALCULATOR AGENT
# ============================================================================

class CalculatorAgent:
    """
    Calculator agent using bind_tools pattern with GPT-4o-mini
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize calculator agent
        
        Args:
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
        """
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Define tools
        self.tools = [add, subtract, multiply, divide]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
        
        print(f"✅ Calculator Agent initialized")
        print(f"   Model: {model}")
        print(f"   Tools: {[tool.name for tool in self.tools]}\n")
    
    def _build_graph(self):
        """Build the agent graph with tool calling"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add agent node
        workflow.add_node("agent", self._call_agent)
        
        # Add tool node
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # After tools execute, return to agent
        workflow.add_edge("tools", "agent")
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _call_agent(self, state: AgentState):
        """
        Agent node: call LLM with tools bound
        """
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def run(self, user_input: str, thread_id: str = "default"):
        """
        Run the calculator agent
        
        Args:
            user_input: User's calculation request
            thread_id: Thread ID for conversation memory
        
        Returns:
            Final response content
        """
        
        print(f"\n📝 Input: {user_input}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        final_message = result["messages"][-1]
        print(f"✅ Answer: {final_message.content}\n")
        
        return final_message.content


# ============================================================================
# 🎯 GRAPH BUILDER FOR LANGGRAPH STUDIO
# ============================================================================

def build_graph():
    """
    Build and return the compiled graph for LangGraph Studio
    This is the main entry point that LangGraph Studio will call
    """
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [add, subtract, multiply, divide]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define agent node
    def call_agent(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools, return to agent
    workflow.add_edge("tools", "agent")

    # Compile without checkpointer (LangGraph Studio handles persistence)
    graph = workflow.compile()

    return graph


# For LangGraph Studio
graph = build_graph()


# ============================================================================
# 🧪 INTERACTIVE MODE
# ============================================================================

def interactive_calculator():
    """Interactive calculator mode"""
    
    # Load API keys
    load_api_keys_from_env()
    
    agent = CalculatorAgent()
    thread_id = "interactive-session"
    
    print("="*70)
    print("INTERACTIVE CALCULATOR")
    print("="*70)
    print("\nExamples:")
    print("  • What is 5 plus 3?")
    print("  • Calculate 100 divided by 4")
    print("  • Multiply 7 and 8, then add 5")
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("💭 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        try:
            agent.run(user_input, thread_id=thread_id)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ============================================================================
# 🚀 MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       LANGGRAPH CALCULATOR AGENT FOR STUDIO                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Run interactive mode
    interactive_calculator()