# Agentic POC — Phase 1 & 2: Tool-Calling, Memory & Multi-Agent Teams

This Proof of Concept (POC) establishes hands-on experience with agentic workflows using LangChain and LangGraph. It is built on top of the existing `AskRef` engineering library to provide high-quality retrieval capabilities to autonomous agents.

## Features
- **Multi-Agent Orchestration (Phase 2)**: Specialized **Librarian** (Research) and **Architect** (Synthesis) agents working in a LangGraph state machine.
- **Multi-Tool Support**: Agents choose between local technical retrieval (AskRef) and global web search (DuckDuckGo).
- **Conversation Memory**: Maintains state across multiple turns using LangGraph's `MemorySaver`.
- **Systematic Tool Use**: Forced priority of external knowledge over LLM training data.
- **M1 Pro Stability**: Optimized threading configuration to prevent OpenMP-related segmentation faults.

## Architecture

### Phase 1: ReAct Agent
Uses a single reason-action loop (`agent.py`).

### Phase 2: Multi-Agent Graph
Uses a specialized `StateGraph` (`graph_agent.py`) with controlled handoffs:
1. **Librarian Node**: Gathering facts via `AskRef` and `search_web`.
2. **Tool Node**: Executing the requested searches.
3. **Architect Node**: Synthesizing raw research into professional reports (ADRs).

## Setup Instructions

### 1. Environment
```bash
cd projects/agentic-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Ensure a `.env` file exists with your `OPENROUTER_API_KEY`. The POC uses `mistralai/mistral-medium-3` via OpenRouter for reliable tool-calling.

### 3. Execution
```bash
# Critical for M1 Pro stability
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Run the Multi-Agent system
python3 graph_agent.py
```

## Step-by-Step implementation History

### Step 1-3 (Phase 1): Base Agent & Tools
- Initialized LangChain/LangGraph ReAct agent.
- Integrated `AskRef` hybrid retrieval and DuckDuckGo web search.

### Step 4 (Phase 2): Multi-Agent Team
- Implemented a `StateGraph` to separate **Research** from **Synthesis**.
- Created a specialized **Librarian** node to maintain data integrity.
- Verified state handoffs where the **Architect** synthesizes evidence gathered by the Librarian.

## Honest CV Claims Unlocked
- "Built a multi-agent orchestration system in LangGraph with specialized Research and Synthesis nodes."
- "Implemented a Librarian/Architect pattern to ensure model integrity using retrieval-as-a-tool."
- "Managed complex state handoffs and tool loops in a deterministic agentic graph."
