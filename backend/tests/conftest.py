import sys
import os
from pathlib import Path

# Force TESTING environment variable
os.environ["TESTING"] = "true"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "mock_cache"
os.environ["FASTEMBED_CACHE_PATH"] = "mock_cache"

# Add backend to Python path
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse

# 1. Monkeypatch QdrantClient __init__ early to force in-memory mode for all tests
original_qdrant_init = QdrantClient.__init__

def mocked_qdrant_client_init(self, *args, **kwargs):
    kwargs["location"] = ":memory:"
    kwargs.pop("url", None)
    kwargs.pop("host", None)
    kwargs.pop("port", None)
    kwargs.pop("grpc_port", None)
    kwargs.pop("path", None)
    original_qdrant_init(self, *args, **kwargs)

QdrantClient.__init__ = mocked_qdrant_client_init

# 2. Monkeypatch HuggingFaceEmbeddings to avoid model load and satisfy Pydantic
def mock_hf_init(self, *args, **kwargs):
    self.__dict__["model_name"] = "BAAI/bge-small-en-v1.5"
    self.__dict__["model_kwargs"] = kwargs.get("model_kwargs", {"device": "cpu"})
    self.__dict__["encode_kwargs"] = kwargs.get("encode_kwargs", {"normalize_embeddings": True})
    self.__dict__["__pydantic_fields_set__"] = {"model_name", "model_kwargs", "encode_kwargs"}
    self.__dict__["__pydantic_extra__"] = {}
    self.__dict__["__pydantic_private__"] = None

def mock_hf_embed_documents(self, texts):
    return [[0.1] * 384 for _ in texts]

def mock_hf_embed_query(self, text):
    return [0.1] * 384

HuggingFaceEmbeddings.__init__ = mock_hf_init
HuggingFaceEmbeddings.embed_documents = mock_hf_embed_documents
HuggingFaceEmbeddings.embed_query = mock_hf_embed_query

# 3. Monkeypatch FastEmbedSparse to avoid model load and satisfy Pydantic
def mock_fe_init(self, *args, **kwargs):
    self.__dict__["model_name"] = "prithivida/Splade_PP_en_v1"
    self.__dict__["cache_dir"] = "mock_cache"
    self.__dict__["__pydantic_fields_set__"] = {"model_name", "cache_dir"}
    self.__dict__["__pydantic_extra__"] = {}
    self.__dict__["__pydantic_private__"] = None

def mock_fe_embed_documents(self, texts):
    return [SparseVector(indices=[1, 2], values=[0.5, 0.5]) for _ in texts]

def mock_fe_embed_query(self, text):
    return SparseVector(indices=[1, 2], values=[0.5, 0.5])

FastEmbedSparse.__init__ = mock_fe_init
FastEmbedSparse.embed_documents = mock_fe_embed_documents
FastEmbedSparse.embed_query = mock_fe_embed_query

# 4. Mock FlashrankRerank before it's called
class MockFlashrankRerank:
    def __init__(self, *args, **kwargs):
        self.top_n = kwargs.get("top_n", 4)
        
    def compress_documents(self, documents, query):
        for i, doc in enumerate(documents):
            doc.metadata["relevance_score"] = 0.9 - (i * 0.1)
        return documents[:self.top_n]

patcher_fr = patch("langchain_community.document_compressors.FlashrankRerank", new=MockFlashrankRerank)
patcher_fr.start()

# Now import application components
from fastapi.testclient import TestClient
from main import app
from app.services.qdrant import qdrant_service
from app.services.llm_service import llm_service

@pytest.fixture(scope="session", autouse=True)
def mock_external_calls():
    yield
    patcher_fr.stop()

@pytest.fixture
def mock_qdrant_client():
    """In-memory client for testing."""
    return QdrantClient()

@pytest.fixture(autouse=True)
def setup_qdrant_service(mock_qdrant_client):
    """Override production Qdrant with in-memory instance."""
    original_client = qdrant_service.client
    original_store = qdrant_service.store
    
    qdrant_service.client = mock_qdrant_client
    # Re-initialize test collection and LangChain store
    qdrant_service._ensure_collection()
    qdrant_service.store = qdrant_service.get_vector_store()
    
    yield qdrant_service
    
    qdrant_service.client = original_client
    qdrant_service.store = original_store

@pytest.fixture
def mock_llm_stream():
    """Mock streaming token generator from vLLM."""
    async def mock_stream(*args, **kwargs):
        tokens = ["This ", "is ", "a ", "mocked ", "response ", "from ", "the ", "support ", "AI.\n", "Sources: [test.txt]"]
        for token in tokens:
            yield token
    
    with patch.object(llm_service, "stream_chat", side_effect=mock_stream) as mock_method:
        yield mock_method

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)
