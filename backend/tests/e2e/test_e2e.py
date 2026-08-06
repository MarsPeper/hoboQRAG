import pytest

def test_complete_rag_pipeline_e2e(client, setup_qdrant_service, mock_llm_stream):
    """End-to-End test validating collection creation, file upload, chunk index, and chat query."""
    service = setup_qdrant_service
    col_name = "e2e_test_collection"
    
    # Cleanup past collection if present
    if service.collection_exists(col_name):
        service.client.delete_collection(col_name)
        
    # 1. Create a test Qdrant collection
    create_resp = client.post("/api/collections", json={"name": col_name})
    assert create_resp.status_code == 200
    assert "created successfully" in create_resp.json()["message"]
    
    # 2. Upload a known test document with specific facts
    doc_content = (
        "The company support portal is located at support.example.com.\n"
        "The default timeout is 30 seconds.\n"
        "Feature X requires version 5.2 or newer."
    )
    file_payload = {"file": ("kb_rules.txt", doc_content.encode("utf-8"), "text/plain")}
    
    upload_resp = client.post(f"/api/collections/{col_name}/documents", files=file_payload)
    assert upload_resp.status_code == 201
    assert upload_resp.json()["chunk_count"] > 0
    
    # 3. Submit a known query targeting this collection
    chat_payload = {
        "prompt": "What is the default timeout?",
        "collection_name": col_name,
        "top_k": 2
    }
    
    chat_resp = client.post("/api/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    assert len(chat_resp.text) > 0
    
    # Clean up test collection
    service.client.delete_collection(col_name)
