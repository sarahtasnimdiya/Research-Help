import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DATA_DIR = "data"
PERSIST_DIR = "storage"

def main():

    # use local embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        print("Index created.")
    else:
        print("Storage already exists. Delete 'storage/' to re-ingest.")

if __name__ == "__main__":
    main()
