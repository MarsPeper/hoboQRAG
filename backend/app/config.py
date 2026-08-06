import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from backend/.env or backend/.env.test if in testing mode
import os
test_env_path = Path(__file__).resolve().parent.parent / ".env.test"
if os.getenv("TESTING") == "true" and test_env_path.exists():
    load_dotenv(dotenv_path=test_env_path, override=True)
else:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

class Settings:
    # FastAPI configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5000))

    # External Service Connections
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_GRPC_URL: str = os.getenv("QDRANT_GRPC_URL", "localhost:6334")
    VLLM_URL: str = os.getenv("VLLM_URL", "http://localhost:8000/v1")

    # Models
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "c:/Projects/hoboQRAG/EmbeddingModels")
    
    # Collections
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "tech_support_kb")

    # Local storage for temp uploads
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "temp_uploads"

    # Ingestion queue & Scheduled ingestion configuration
    DOCUMENT_SOURCE_DIR: Path = Path(os.getenv("DOCUMENT_SOURCE_DIR", "c:/Projects/hoboQRAG/knowledge_base"))
    INGESTION_SCHEDULE: str = os.getenv("INGESTION_SCHEDULE", "00:00")
    INGESTION_GPU_MAINTENANCE: str = os.getenv("INGESTION_GPU_MAINTENANCE", "true").lower()
    VLLM_CONTAINER_NAME: str = os.getenv("VLLM_CONTAINER_NAME", "vllm-server")
    QUEUE_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "queue.db"
    QUEUE_FILES_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "queue_files"

    # Tech Support System Instructions
    SYSTEM_INSTRUCTION: str = (
        "You are an expert internal technical support AI assistant. Your goal is to help support agents "
        "resolve customer issues quickly and accurately using only the provided reference documents.\n\n"
        "RULES:\n"
        "1. FACTUALITY: Rely strictly on the provided context. If the answer cannot be found in the context, "
        "say 'I cannot find the solution in the local knowledge base' and do not make up steps.\n"
        "2. SOURCE CITATION: At the end of your response, list the source files and pages used to answer the "
        "question (e.g., 'Sources: [KB-102_Router_Setup.pdf]').\n"
        "3. FORMATTING: Present troubleshooting guides in numbered, chronological steps. Format CLI commands, "
        "paths, and registry keys in code blocks (e.g., `ping 8.8.8.8`).\n"
        "4. DIRECTNESS: Do not write conversational pleasantries. Provide the solution immediately."
    )

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCUMENT_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
settings.QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.QUEUE_FILES_DIR.mkdir(parents=True, exist_ok=True)
