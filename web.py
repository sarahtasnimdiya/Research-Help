import os
import uuid
import zipfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP


# Config

BASE_PROJECTS_DIR = "uploaded_projects"
BASE_STORAGE_DIR = "project_storage"
MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

os.makedirs(BASE_PROJECTS_DIR, exist_ok=True)
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)


# App

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Local models (ONLY local)

llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=512,
    context_window=4096,
    verbose=False,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# force llama-index to use your local LLM
Settings.llm = llm


# Helpers

def save_and_prepare_project(zip_path: str) -> str:
    project_id = str(uuid.uuid4())

    project_dir = os.path.join(BASE_PROJECTS_DIR, project_id)
    storage_dir = os.path.join(BASE_STORAGE_DIR, project_id)

    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)

    # unzip project
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(project_dir)

    # read files
    documents = SimpleDirectoryReader(
        project_dir,
        recursive=True
    ).load_data()

    # build index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=storage_dir)

    return project_id


def query_project(project_id: str, question: str) -> str:

    storage_dir = os.path.join(BASE_STORAGE_DIR, project_id)

    if not os.path.exists(storage_dir):
        return "Project not found."

    storage_context = StorageContext.from_defaults(
        persist_dir=storage_dir
    )

    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=4)

    response = query_engine.query(question)

    return str(response)



# Routes

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/upload")
async def upload_project(file: UploadFile = File(...)):

    if not file.filename.endswith(".zip"):
        return JSONResponse(
            {"error": "Please upload a ZIP file"},
            status_code=400
        )

    tmp_zip = f"tmp_{uuid.uuid4().hex}.zip"

    with open(tmp_zip, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        project_id = save_and_prepare_project(tmp_zip)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    return {"project_id": project_id}


@app.post("/ask")
async def ask(
    project_id: str = Form(...),
    question: str = Form(...)
):
    answer = query_project(project_id, question)
    return {"answer": answer}
