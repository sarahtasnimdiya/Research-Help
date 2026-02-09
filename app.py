import os
import asyncio

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.llama_cpp import LlamaCPP

from tools import scan_project


PERSIST_DIR = "storage"
MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# Global cached agent
_AGENT = None


def get_agent():
    """
    Create (once) and return the local ReAct agent.
    This function is safe to reuse from CLI and Web.
    """

    global _AGENT

    if _AGENT is not None:
        return _AGENT

    # ---- LLM (local only)
    llm = LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=512,
        context_window=4096,
        verbose=False,
    )

    # IMPORTANT: prevent OpenAI fallback
    Settings.llm = llm

    # ---- Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Load stored index
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR
    )

    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=4)

    # ---- Tool

    def scan_my_project():
        """
        Scan the current project directory automatically.
        """
        return scan_project(PROJECT_ROOT)

    scan_tool = FunctionTool.from_defaults(
        fn=scan_my_project,
        name="scan_project",
        description="Scan and list all files in the current project directory."
    )

    # ---- Agent
    _AGENT = ReActAgent(
        tools=[scan_tool],
        llm=llm,
        retriever=retriever,
        verbose=True,
    )

    return _AGENT

# CLI interface
async def main():

    agent = get_agent()

    print("Local project assistant agent is ready. Type 'exit' to stop.")

    while True:
        query = input("\n>> ")

        if query.lower() == "exit":
            break

        result = await agent.run(query)
        print("\n", result)


if __name__ == "__main__":
    asyncio.run(main())
