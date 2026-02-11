from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import MODEL_PATH


def build_query_engine(project_path: Path):


    # Embeddings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Local LLM (more deterministic)
    llm = LlamaCPP(
        model_path=str(MODEL_PATH),
        temperature=0.0,   # IMPORTANT: prevent guessing
        max_new_tokens=512,
        context_window=4096,
        n_gpu_layers=35,
        verbose=False
    )


    # Chunking (keep your good setup)
    Settings.node_parser = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load project files
    docs = SimpleDirectoryReader(
        input_dir=str(project_path),
        recursive=True,
        filename_as_id=True,
        required_exts=[
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
            ".md", ".txt", ".html", ".css", ".json"
        ]
    ).load_data()


    # Build index
    index = VectorStoreIndex.from_documents(docs)

    # Retriever
    retriever = index.as_retriever(
        similarity_top_k=6
    )

    
    # STRONG anti-hallucination prompt
    qa_prompt = PromptTemplate(
        """You are an assistant helping a user understand an uploaded software project.

You must answer ONLY using the information present in the context.

If the answer cannot be directly found in the context, reply exactly with:
I could not find this information in the uploaded project.

Do NOT guess.
Do NOT use outside knowledge.
Do NOT infer or assume missing information.

Context:
---------------------
{context_str}
---------------------

Question:
{query_str}

Answer (use only the context):
"""
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=qa_prompt
    )

    return query_engine
