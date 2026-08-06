import pytest
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock

def test_create_and_list_collections(setup_qdrant_service):
    """Test creating collections, checking existence, and avoiding duplicate creation."""
    service = setup_qdrant_service
    
    # Test checking default collection exists
    assert service.collection_exists(service.collection_name) is True
    
    # Test creating new collection
    test_col = "another_collection"
    created = service.create_collection(test_col)
    assert created is True
    assert service.collection_exists(test_col) is True
    
    # Test creating duplicate collection
    created_duplicate = service.create_collection(test_col)
    assert created_duplicate is False

def test_collection_not_found(setup_qdrant_service):
    """Test checking collection that does not exist."""
    service = setup_qdrant_service
    assert service.collection_exists("non_existent_collection") is False

def test_insert_and_search_vectors(setup_qdrant_service):
    """Test inserting documents with metadata, and running hybrid similarity searches."""
    service = setup_qdrant_service
    docs = [
        Document(page_content="The server address is 10.0.0.1", metadata={"file_name": "network.txt", "page": 1}),
        Document(page_content="Admin password is admin123", metadata={"file_name": "secrets.txt", "page": 2})
    ]
    service.insert_documents(docs)
    
    # Verify vector search retrieves relevant documentation
    results = service.search_hybrid("server address", limit=2)
    assert len(results) > 0
    filenames = [doc.metadata["file_name"] for doc in results]
    assert "network.txt" in filenames
    assert "base_score" in results[0].metadata

def test_delete_file(setup_qdrant_service):
    """Test deleting point documents from collections by matching metadata."""
    service = setup_qdrant_service
    docs = [
        Document(page_content="Chunk of file A", metadata={"file_name": "file_a.txt", "page": 1}),
        Document(page_content="Chunk of file B", metadata={"file_name": "file_b.txt", "page": 1})
    ]
    service.insert_documents(docs)
    
    # Verify unique file list counts
    files = service.list_unique_files()
    assert len(files) == 2
    
    # Delete points for file_a.txt
    success = service.delete_file("file_a.txt")
    assert success is True
    
    # Verify only file_b.txt remains
    files = service.list_unique_files()
    assert len(files) == 1
    assert files[0]["file_name"] == "file_b.txt"

def test_qdrant_unavailable_error(setup_qdrant_service):
    """Test error handling when Qdrant is disconnected."""
    service = setup_qdrant_service
    with patch.object(service.client, "get_collections", side_effect=Exception("Qdrant connection refused")):
        with pytest.raises(Exception) as excinfo:
            service.collection_exists("any_col")
        assert "connection refused" in str(excinfo.value).lower()
