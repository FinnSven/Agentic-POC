import os
import sys
from pathlib import Path
import yaml

# Add askref to path
ASKREF_ROOT = Path(__file__).parent.parent / "askref"
sys.path.append(str(ASKREF_ROOT))

from askref.embedder import load_model
from askref.index_manager import load_index
from askref.retriever import retrieve
from langchain_core.tools import tool

class AskRefTool:
    def __init__(self):
        # Load config
        config_path = ASKREF_ROOT / "config.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Override directory to absolute path (if it uses ~)
        self.config["index"]["directory"] = os.path.expanduser(self.config["index"]["directory"])
        self.config["library"]["path"] = os.path.expanduser(self.config["library"]["path"])
        
        index_dir = Path(self.config["index"]["directory"])
        
        # Load index and model
        print(f"Loading AskRef index from {index_dir}...")
        self.index, self.db = load_index(index_dir)
        
        model_name = self.config.get("models", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"Loading embedding model: {model_name}...")
        self.model = load_model(model_name)

    def search(self, query: str, category: str = None) -> str:
        """Search the engineering library for technical information."""
        chunks = retrieve(
            query, self.index, self.db, self.model, self.config,
            category_filter=category
        )
        
        if not chunks:
            return "No matching information found in the engineering library."
        
        # Format results for the agent
        formatted_results = []
        for i, chunk in enumerate(chunks):
            # The retrieve function returns a list of RetrievedChunk dataclass objects.
            source = chunk.source_path
            text = chunk.text_preview
            title = chunk.title
            formatted_results.append(f"Result {i+1} from '{title}' ({source}):\n{text}\n")
            
        return "\n---\n".join(formatted_results)

# Initialize singleton
_askref_instance = None

@tool
def search_engineering_library(query: str):
    """
    Search the engineering library (534+ books) for high-quality technical information.
    Use this for questions about architecture, .NET, Java, Python, AI, databases, etc.
    """
    global _askref_instance
    if _askref_instance is None:
        _askref_instance = AskRefTool()
    return _askref_instance.search(query)
