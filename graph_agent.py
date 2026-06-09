import os
import operator
from typing import Annotated, List, TypedDict, Union

# Fix threading crash on M1 Pro
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Local tools
from askref_tool import search_engineering_library
from agent import search_web, get_current_time

load_dotenv()

# --- CONFIGURATION ---
llm = ChatOpenAI(
    model="mistralai/mistral-medium-3",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/FinnSven/agentic-poc",
        "X-Title": "Agentic POC",
    }
)

tools = [search_engineering_library, search_web, get_current_time]
tool_node = ToolNode(tools)

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    # The 'operator.add' allows us to append messages rather than overwrite
    messages: Annotated[List[BaseMessage], operator.add]
    research_summary: str
    final_report: str

# --- NODE: LIBRARIAN (The Researcher) ---
def librarian_node(state: AgentState):
    """
    The Librarian's job is to gather all necessary facts using tools.
    It focuses ONLY on retrieval and accuracy.
    """
    system_msg = (
        "You are the Lead Librarian. Your job is to RESEARCH a topic thoroughly. "
        "Use search_engineering_library for technical facts and search_web for news/context. "
        "You MUST find concrete evidence before passing the task to the Architect. "
        "Tool outputs are tagged at the source: book results begin with [LIBRARY SOURCE], "
        "web results begin with [WEB SOURCE]. PRESERVE THIS PROVENANCE. "
        "When you summarize, use exactly two headings and attribute each finding to its origin: "
        "'LIBRARY FINDINGS (from search_engineering_library)' listing only [LIBRARY SOURCE] book "
        "facts (name the book), and 'WEB FINDINGS (from search_web)' listing only [WEB SOURCE] facts "
        "(name the URL). NEVER move a web finding under the library heading or vice versa. "
        "If a search returned nothing relevant, say so under that heading rather than filling it. "
        "When done, end your summary with 'RESEARCH_COMPLETE'."
    )
    
    # We prefix the messages with the Librarian's system prompt
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    
    # Check if we already have tool results in the message history to decide what to do
    # For a simple sequential graph, we just let it call tools.
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

# --- NODE: ARCHITECT (The Synthesizer) ---
def architect_node(state: AgentState):
    """
    The Architect takes the research and structures it into a professional response.
    It does NOT use tools; it relies on the Librarian's work.
    """
    # Extract research findings from the Librarian's messages
    research_context = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, (HumanMessage, SystemMessage)): continue
        if "RESEARCH_COMPLETE" in str(msg.content):
            research_context = msg.content
            break
    
    if not research_context:
        # Fallback: take the last non-tool message
        research_context = state["messages"][-1].content

    system_msg = (
        "You are the Lead System Architect. Your job is to take the Librarian's research "
        "and produce a professional, structured report or architectural recommendation. "
        "Strictly adhere to the facts provided by the Librarian and PRESERVE SOURCE PROVENANCE. "
        "Section 2 'Technical Depth (AskRef)' must cite ONLY the Librarian's LIBRARY FINDINGS "
        "(books from the engineering library) — cite each book by name. "
        "Section 3 'Context (Web)' must cite ONLY the Librarian's WEB FINDINGS (URLs). "
        "NEVER present a web URL as a library/AskRef source, and never relabel across sections. "
        "If the Librarian found no library sources, write 'No engineering-library sources were "
        "retrieved for this query.' under Section 2 — do NOT backfill it with web content. "
        "Begin the report with the title line '**Report:' so it is identifiable. "
        "Structure: 1. Overview, 2. Technical Depth (AskRef), 3. Context (Web), 4. Recommendation."
    )
    
    prompt = f"Based on this research:\n{research_context}\n\nFinalize the report for the user."
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
    
    return {"final_report": response.content, "messages": [response]}

# --- GRAPH LOGIC ---
def should_continue(state: AgentState):
    """Determines if the Librarian needs to call more tools or move to the Architect."""
    last_message = state["messages"][-1]
    
    # If the LLM wants to call a tool, go to 'tools'
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # If the Librarian said it's done, go to 'architect'
    if "RESEARCH_COMPLETE" in str(last_message.content):
        return "architect"
    
    # Otherwise, it might be just a chat response from librarian (unlikely with our prompt)
    return "architect"

# --- COMPILATION ---
workflow = StateGraph(AgentState)

workflow.add_node("librarian", librarian_node)
workflow.add_node("tools", tool_node)
workflow.add_node("architect", architect_node)

workflow.add_edge(START, "librarian")
workflow.add_conditional_edges("librarian", should_continue, ["tools", "architect"])
workflow.add_edge("tools", "librarian") # Tool loop
workflow.add_edge("architect", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- EXECUTION ---
if __name__ == "__main__":
    print("--- STARTING MULTI-AGENT POC (PHASE 2) ---")
    config = {"configurable": {"thread_id": "multi_agent_1"}}
    
    query = (
        "Should we use Event Sourcing for a high-frequency trading system? "
        "Search the engineering library for pros/cons and the web for any recent "
        "post-mortems or articles from 2025/2026 regarding this pattern."
    )
    
    print(f"\nUser: {query}\n")
    
    for event in app.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode="values"):
        if "messages" in event:
            msg = event["messages"][-1]
            if not isinstance(msg, HumanMessage):
                node_name = "Agent"
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    node_name = "ToolCall"
                elif "RESEARCH_COMPLETE" in str(msg.content):
                    node_name = "Librarian"
                elif str(msg.content).startswith("**Report"):
                    node_name = "Architect"
                
                print(f"[{node_name}]: {str(msg.content)[:100]}...")

    final_state = app.get_state(config)
    print("\n" + "="*50)
    print("FINAL ARCHITECT REPORT:")
    print("="*50)
    print(final_state.values.get("final_report", "No report generated."))
