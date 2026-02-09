from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "mistral-7b-instruct.Q4_K_M.gguf"

UPLOAD_DIR = BASE_DIR / "data" / "uploaded_projects"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
