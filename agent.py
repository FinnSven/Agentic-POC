import os

# Fix threading crash on M1 Pro
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from duckduckgo_search import DDGS
from askref_tool import search_engineering_library

load_dotenv()

# Setup OpenRouter
llm = ChatOpenAI(
    model="mistralai/mistral-medium-3",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/FinnSven/agentic-poc",
        "X-Title": "Agentic POC",
    }
)

@tool
def get_current_time():
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_web(query: str):
    """
    Search the internet for current events, news, or general information not found in the engineering library.
    Use this for information that changes frequently (stock prices, today's news) or non-technical topics.
    """
    print(f"DEBUG: search_web calling with '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                print("DEBUG: search_web returned NO results")
                return "No web results found."
            
            print(f"DEBUG: search_web found {len(results)} results")
            formatted = []
            for i, r in enumerate(results):
                formatted.append(f"Result {i+1} from {r['href']}:\n{r['body']}\n")
            return "\n---\n".join(formatted)
    except Exception as e:
        print(f"DEBUG: search_web error: {e}")
        return f"Error searching the web: {e}"

tools = [get_current_time, search_engineering_library, search_web]

# Add memory
checkpointer = MemorySaver()

from langgraph.prebuilt import create_react_agent

# Add a system message to encourage tool use
system_prompt = (
    "You are a professional software architect. "
    "You MUST use search_engineering_library for any technical questions about software, architecture, or languages. "
    "You MUST use search_web for current events, people, or news. "
    "Do not rely on your internal training data if a tool is available."
)

agent_executor = create_react_agent(llm, tools, checkpointer=checkpointer, prompt=system_prompt)

if __name__ == "__main__":
    print("--- STARTING AGENTIC POC SESSION 3 ---")
    
    config = {"configurable": {"thread_id": "session_1"}}
    
    # Turn 1: Library Search (Specific to your library)
    q1 = "Search the engineering library for the definition of 'Ubiquitous Language' and tell me which book it comes from."
    print(f"\nUser: {q1}")
    for chunk in agent_executor.stream({"messages": [("user", q1)]}, config, stream_mode="values"):
        message = chunk["messages"][-1]
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"Tool Call: {tc['name']}({tc['args']})")
    
    print(f"Agent: {chunk['messages'][-1].content}")
    
    # Turn 2: Web Search + Memory
    q2 = "Who is the current Prime Minister of Sweden? Also, compare their stance on technology to what you just told me about DDD."
    print(f"\nUser: {q2}")
    for chunk in agent_executor.stream({"messages": [("user", q2)]}, config, stream_mode="values"):
        message = chunk["messages"][-1]
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"Tool Call: {tc['name']}({tc['args']})")
    
    print(f"Agent: {chunk['messages'][-1].content}")
