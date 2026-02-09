from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import MODEL_PATH

def build_query_engine(project_path):

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = LlamaCPP(
        model_path=str(MODEL_PATH),
        temperature=0.1,
        max_new_tokens=256,
        context_window=4096,
        n_gpu_layers=35,
        verbose=False
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    docs = SimpleDirectoryReader(
        input_dir=str(project_path),
        recursive=True,
        filename_as_id=True
    ).load_data()

    index = VectorStoreIndex.from_documents(docs)

    return index.as_query_engine(similarity_top_k=4)
