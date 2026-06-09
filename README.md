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

## Verification & fixes (2026-06-09)
Both phases were re-run end-to-end and the output inspected (not trusted from logs). Two real defects were found and fixed:

1. **Web search was dead.** `duckduckgo_search` is deprecated (renamed to `ddgs`); `search_web` silently returned no results, so the web leg of both phases produced nothing. Fixed by switching to `ddgs`. Re-verified: the agent now retrieves current web results (e.g. correctly identified the sitting Prime Minister of Sweden).
2. **Source-provenance contamination in Phase 2.** The Architect was placing *web* citations under the "Technical Depth (AskRef)" heading and dropping the actual retrieved book — i.e. mislabelling web sources as library sources, the exact failure the Librarian/Architect split is meant to prevent. Fixed at three levels: tools now stamp output with `[LIBRARY SOURCE]` / `[WEB SOURCE]`; the Librarian must summarise under separate, attributed headings; the Architect must keep library and web citations in their own sections and never relabel. Re-verified: Section 2 now cites only real books (Financial Data Engineering, Developing High-Frequency Trading Systems, Clean Code with C# 2nd Ed, Software Architecture Patterns for Serverless Systems — all confirmed present in the library) and Section 3 only URLs.

## Known limitations (be honest in demos)
- **Web-source quality is not scored.** `ddgs` returns whatever ranks; a run surfaced low-quality `luxoret.com/prompts/ai-chat/...` pages. Provenance is separated and labelled, but quality is not yet filtered. Next hardening step.
- It is a **terminal-run POC**, not a deployed service. No Streamlit/web front-end yet (Phase 3, not built).
- The provenance guarantee is **prompt-enforced + source-tagged**, not schema-enforced; a sufficiently contrary model could still err. A typed `research_data` state field per source would harden it further.

## Honest CV Claims Unlocked (verified by running, 2026-06-09)
- "Built a multi-agent orchestration system in LangGraph with specialized Research (Librarian) and Synthesis (Architect) nodes."
- "Implemented source-provenance separation so library and web evidence are tagged and never conflated — found and fixed a contamination bug where the synthesizer mislabelled web sources as library sources."
- "Managed state handoffs and tool loops in a LangGraph agentic graph, with retrieval-as-a-tool over a 1080-book library (AskRef) plus live web search."
- Do NOT claim: production deployment, a web UI, or validated web-source quality.
