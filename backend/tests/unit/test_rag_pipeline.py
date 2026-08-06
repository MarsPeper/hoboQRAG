import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

@pytest.mark.asyncio
async def test_pipeline_orchestration_success(client, setup_qdrant_service, mock_llm_stream):
    """Test standard pipeline flow: Search -> Rerank -> LLM Generation."""
    service = setup_qdrant_service
    mock_docs = [Document(page_content="RAG context info", metadata={"file_name": "kb.txt", "page": 1})]
    
    with patch.object(service, "search_hybrid", return_value=mock_docs) as mock_search, \
         patch.object(service, "rerank", return_value=mock_docs) as mock_rerank:
         
        response = client.post("/api/chat", json={
            "prompt": "How does RAG work?",
            "collection_name": service.collection_name,
            "top_k": 2
        })
        
        assert response.status_code == 200
        assert "mocked response from the support AI" in response.text
        
        # Verify orchestration stages are called with proper arguments
        mock_search.assert_called_once_with("How does RAG work?", limit=15, collection_name=service.collection_name)
        mock_rerank.assert_called_once_with(query="How does RAG work?", documents=mock_docs, top_n=2)

@pytest.mark.asyncio
async def test_pipeline_no_results(client, setup_qdrant_service, mock_llm_stream):
    """Test pipeline response when no documents match the search."""
    service = setup_qdrant_service
    with patch.object(service, "search_hybrid", return_value=[]) as mock_search:
        response = client.post("/api/chat", json={
            "prompt": "Unrelated topic query",
            "collection_name": service.collection_name
        })
        assert response.status_code == 200
        # Should still output AI stream response
        assert "mocked response" in response.text

@pytest.mark.asyncio
async def test_pipeline_qdrant_failure(client, setup_qdrant_service):
    """Test error handling when the vector store fails to search."""
    service = setup_qdrant_service
    with patch.object(service, "search_hybrid", side_effect=Exception("Qdrant connection timeout")):
        response = client.post("/api/chat", json={
            "prompt": "What is up?",
            "collection_name": service.collection_name
        })
        assert response.status_code == 500
        assert "timeout" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_pipeline_reranker_failure_fallback(client, setup_qdrant_service, mock_llm_stream):
    """Test that reranker failure falls back gracefully without breaking the pipeline."""
    service = setup_qdrant_service
    mock_docs = [Document(page_content="Fallback context data", metadata={"file_name": "kb.txt", "page": 1})]
    
    with patch.object(service, "search_hybrid", return_value=mock_docs), \
         patch("app.services.qdrant.FlashrankRerank", side_effect=Exception("Reranker GPU OOM")):
         
        response = client.post("/api/chat", json={
            "prompt": "How does RAG work?",
            "collection_name": service.collection_name
        })
        assert response.status_code == 200
        assert "mocked response" in response.text

@pytest.mark.asyncio
async def test_pipeline_invalid_collection(client):
    """Test API response when requesting an invalid/non-existent collection."""
    response = client.post("/api/chat", json={
        "prompt": "Hello",
        "collection_name": "non_existent_collection_name"
    })
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_pipeline_empty_question(client, setup_qdrant_service):
    """Test validation reject for empty question prompt."""
    response = client.post("/api/chat", json={
        "prompt": "",
        "collection_name": setup_qdrant_service.collection_name
    })
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()
