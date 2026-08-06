import pytest
from unittest.mock import patch, MagicMock

def test_api_list_collections(client, setup_qdrant_service):
    """Test retrieving list of collections."""
    response = client.get("/api/collections")
    assert response.status_code == 200
    assert "collections" in response.json()
    assert setup_qdrant_service.collection_name in response.json()["collections"]

def test_api_create_collection_success(client, setup_qdrant_service):
    """Test successful collection creation."""
    col_name = "new_kb_collection"
    response = client.post("/api/collections", json={"name": col_name})
    assert response.status_code == 200
    assert "created successfully" in response.json()["message"]
    
    # Verify it is listed now
    list_resp = client.get("/api/collections")
    assert col_name in list_resp.json()["collections"]

def test_api_create_collection_duplicate(client, setup_qdrant_service):
    """Test duplicate collection creation block."""
    col_name = setup_qdrant_service.collection_name
    response = client.post("/api/collections", json={"name": col_name})
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()

def test_api_upload_document_success(client, setup_qdrant_service):
    """Test successful document upload, parsing, chunking, and insertion."""
    col = setup_qdrant_service.collection_name
    file_content = b"This is a valid support document content. Very descriptive facts."
    file = {"file": ("test_doc.txt", file_content, "text/plain")}
    
    response = client.post(f"/api/collections/{col}/documents", files=file)
    assert response.status_code == 201
    assert "successfully parsed and indexed file" in response.json()["message"].lower()
    assert response.json()["chunk_count"] > 0

def test_api_upload_document_unsupported(client, setup_qdrant_service):
    """Test upload error on unsupported file formats."""
    col = setup_qdrant_service.collection_name
    file = {"file": ("image.png", b"fake binary data", "image/png")}
    response = client.post(f"/api/collections/{col}/documents", files=file)
    assert response.status_code == 400
    assert "unsupported file extension" in response.json()["detail"].lower()

def test_api_upload_document_empty(client, setup_qdrant_service):
    """Test upload rejection on empty files."""
    col = setup_qdrant_service.collection_name
    file = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post(f"/api/collections/{col}/documents", files=file)
    assert response.status_code == 400
    assert "empty file upload" in response.json()["detail"].lower()

def test_api_upload_document_invalid_collection(client):
    """Test upload rejection on invalid collection targets."""
    file = {"file": ("test.txt", b"content data", "text/plain")}
    response = client.post("/api/collections/invalid_collection_name_123/documents", files=file)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
