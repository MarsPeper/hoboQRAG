import os
import pytest
import httpx
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from app.config import settings

def test_embedding_model_integration():
    """Verify loading the real dense embedding model and generating a vector."""
    try:
        model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
        vec = model.embed_query("test query")
        assert len(vec) == 384
    except Exception as e:
        pytest.fail(f"Embedding model integration failed: {e}")

def test_reranker_integration():
    """Verify loading the real reranker model and reranking documents."""
    try:
        reranker = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=1,
            cache_dir=settings.EMBEDDING_CACHE_DIR
        )
        docs = [
            Document(page_content="Paris is in France"),
            Document(page_content="Apples are green")
        ]
        res = reranker.compress_documents(docs, "Where is Paris?")
        assert len(res) == 1
        assert "France" in res[0].page_content
    except Exception as e:
        pytest.fail(f"Reranker integration failed: {e}")

def test_qdrant_integration():
    """Verify real Qdrant connection and basic collection manipulations."""
    # Connect to the test Qdrant port
    test_client = QdrantClient(url="http://localhost:6343")
    try:
        test_client.get_collections()
    except Exception:
        pytest.skip("Test Qdrant container is not running on localhost:6343")

    col_name = "integration_test_collection"
    try:
        # Cleanup past collection if exists
        collections = test_client.get_collections().collections
        if any(c.name == col_name for c in collections):
            test_client.delete_collection(col_name)
            
        from qdrant_client.models import VectorParams, Distance
        test_client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        collections = test_client.get_collections().collections
        assert any(c.name == col_name for c in collections)
        
        # Cleanup
        test_client.delete_collection(col_name)
    except Exception as e:
        pytest.fail(f"Qdrant integration test failed: {e}")

def test_vllm_integration():
    """Verify connectivity and health of local vLLM / Ollama server."""
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    try:
        response = httpx.get(f"{vllm_url}/models", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"vLLM not responding on {vllm_url} (HTTP {response.status_code})")
    except Exception:
        pytest.skip(f"vLLM is not running or reachable on {vllm_url}")
