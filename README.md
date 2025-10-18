# LangGraph Calculator Agent

A simple calculator agent built with LangGraph and OpenAI, demonstrating tool-calling patterns with LangGraph Studio support.

## Features

- Basic arithmetic operations (add, subtract, multiply, divide)
- Natural language interface
- LangGraph Studio integration
- Interactive command-line mode
- Memory/conversation history support

## Prerequisites

- Python 3.11+
- OpenAI API key
- LangGraph CLI (for Studio)

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd calculator_agentic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

## Usage

### Option 1: LangGraph Studio (Recommended)

1. Start the LangGraph development server:
```bash
langgraph dev
```

2. Open LangGraph Studio in your browser:
   - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
   - API: http://127.0.0.1:2024

3. Start chatting with the calculator agent!

### Option 2: Interactive Command-Line Mode

Run the agent directly in your terminal:
```bash
python calculator_agent.py
```

## Example Queries

- "What is 5 plus 3?"
- "Calculate 100 divided by 4"
- "Multiply 7 and 8, then add 5"
- "What's 15% of 200?"

## Project Structure

```
calculator_agentic/
├── calculator_agent.py   # Main agent implementation
├── langgraph.json       # LangGraph configuration
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variable template
├── .env               # Your API keys (not in git)
└── README.md          # This file
```

## How It Works

The calculator agent uses:
- **LangGraph** for orchestrating the agent workflow
- **OpenAI GPT-4o-mini** as the language model
- **Tool calling** pattern to execute calculations
- **State management** for conversation history

The agent can understand natural language math queries and automatically calls the appropriate calculator tools to perform operations.

## Development

### Graph Structure

The agent follows this workflow:
1. User input → Agent node
2. Agent decides if tools are needed
3. If yes: Execute tools → Return to agent
4. If no: Return final answer

### Adding New Tools

To add new mathematical operations:

1. Define a new tool function with the `@tool` decorator
2. Add it to the `tools` list in `build_graph()`

Example:
```python
@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent
```

## Troubleshooting

**Issue**: "Failed to initialize Studio" or "TypeError: Failed to fetch"
- **Solution**: Make sure `langgraph dev` is running

**Issue**: "OPENAI_API_KEY: Not found"
- **Solution**: Check your `.env` file contains a valid OpenAI API key

**Issue**: Graph fails to load with checkpointer error
- **Solution**: The graph should compile without a custom checkpointer for LangGraph Studio

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
