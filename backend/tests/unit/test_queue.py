import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.services.queue_db import queue_db
from app.services.sync import run_queue_ingestion, DockerManager, ingestion_status
from app.services.qdrant import qdrant_service
from app.config import settings

@pytest.fixture(autouse=True)
def clean_queue_database():
    """Wipes the SQLite database queue tables before each test."""
    conn = queue_db._get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM ingestion_queue")
            conn.execute("DELETE FROM ingestion_history")
    finally:
        conn.close()

def test_enqueue_and_cancel_operation():
    """Test enqueuing and then cancelling an operation."""
    op_id = queue_db.insert_queue_item(
        collection="tech_support_kb",
        document_id="doc1",
        operation="ADD",
        filename="doc1.pdf",
        filepath="dummy/path",
        queued_by="tester"
    )
    
    assert op_id is not None
    item = queue_db.get_queue_item(op_id)
    assert item["document_id"] == "doc1"
    assert item["status"] == "pending"
    
    # Cancel/Delete
    queue_db.delete_queue_item(op_id)
    assert queue_db.get_queue_item(op_id) is None

def test_conflict_resolution_rules():
    """Test conflict resolution rules for enqueued items."""
    # 1. Test ADD conflict
    queue_db.insert_queue_item("col", "doc_add", "ADD", "file.pdf", "path", "user")
    with pytest.raises(ValueError, match="already has a pending ADD"):
        queue_db.resolve_conflicts_and_enqueue("col", "doc_add", "ADD", "file.pdf", "path", "user")

    # 2. Test ADD + DELETE = cancel pending ADD (net zero)
    res = queue_db.resolve_conflicts_and_enqueue("col", "doc_add", "DELETE", "", "", "user")
    assert res == "CANCELLED_PENDING_ADD"
    assert queue_db.get_pending_by_document("doc_add", "col") is None

    # 3. Test REPLACE update
    op_id = queue_db.resolve_conflicts_and_enqueue("col", "doc_rep", "REPLACE", "file.pdf", "path1", "user")
    # Duplicate REPLACE should overwrite path
    op_id_2 = queue_db.resolve_conflicts_and_enqueue("col", "doc_rep", "REPLACE", "file2.pdf", "path2", "user")
    assert op_id == op_id_2
    item = queue_db.get_queue_item(op_id)
    assert item["filename"] == "file2.pdf"
    assert item["filepath"] == "path2"

@pytest.mark.asyncio
async def test_replacement_safety_on_failure():
    """Test that Qdrant's old document versions are not deleted if the replacement insertion fails."""
    # Create a real temp file so exists() returns True
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_filepath = f.name

    try:
        # Enqueue a REPLACE
        op_id = queue_db.insert_queue_item(
            collection="tech_support_kb",
            document_id="doc_failed",
            operation="REPLACE",
            filename="doc.pdf",
            filepath=temp_filepath,
            queued_by="tester"
        )
        
        # Mock QdrantService behavior to throw on insert
        with patch("app.services.sync.settings.INGESTION_GPU_MAINTENANCE", "false"), \
             patch("app.services.sync.DocumentParser.load_file", return_value=[Document(page_content="content")]), \
             patch("app.services.sync.qdrant_service.get_max_document_version", return_value=1), \
             patch("app.services.sync.qdrant_service.insert_documents", side_effect=RuntimeError("Qdrant Down")), \
             patch("app.services.sync.qdrant_service.delete_previous_versions") as mock_delete:
             
             summary = await run_queue_ingestion()
             assert summary["failed"] == 1
             
             # Ensure delete was never called, keeping the original version intact
             mock_delete.assert_not_called()
             
             # Ensure queue item is flagged as failed in SQLite
             item = queue_db.get_queue_item(op_id)
             assert item["status"] == "failed"
             assert "Qdrant Down" in item["error"]
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
            except Exception:
                pass

@pytest.mark.asyncio
async def test_end_to_end_ingestion_success(setup_qdrant_service):
    """E2E simulation: Queue ADD -> Process -> Verify -> Queue REPLACE -> Process -> Verify -> Queue DELETE -> Process -> Verify."""
    service = setup_qdrant_service
    col = service.collection_name
    doc_id = "e2e_manual_test_doc"
    
    # Setup mock file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Initial draft documentation content")
        temp_filepath = f.name
        
    try:
        # 1. Queue ADD
        op_id_1 = queue_db.insert_queue_item(col, doc_id, "ADD", "test_doc.txt", temp_filepath, "tester")
        
        # Process Ingestion
        with patch("app.services.sync.settings.INGESTION_GPU_MAINTENANCE", "false"):
            summary = await run_queue_ingestion()
            assert summary["success"] == 1
            
        # Verify document exists in Qdrant (version 1)
        assert service.verify_version_exists(doc_id, 1, collection_name=col)
        assert service.get_max_document_version(doc_id, collection_name=col) == 1
        
        # 2. Queue REPLACE
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f2:
            f2.write(b"Updated documentation content")
            temp_filepath_2 = f2.name
            
        try:
            op_id_2 = queue_db.insert_queue_item(col, doc_id, "REPLACE", "test_doc_v2.txt", temp_filepath_2, "tester")
            
            with patch("app.services.sync.settings.INGESTION_GPU_MAINTENANCE", "false"):
                summary = await run_queue_ingestion()
                assert summary["success"] == 1
                
            # Verify version 2 exists
            assert service.verify_version_exists(doc_id, 2, collection_name=col)
            # Verify version 1 was purged
            assert not service.verify_version_exists(doc_id, 1, collection_name=col)
            assert service.get_max_document_version(doc_id, collection_name=col) == 2
            
        finally:
            if os.path.exists(temp_filepath_2):
                try: os.unlink(temp_filepath_2)
                except Exception: pass
                
        # 3. Queue DELETE
        op_id_3 = queue_db.insert_queue_item(col, doc_id, "DELETE", "", "", "tester")
        
        with patch("app.services.sync.settings.INGESTION_GPU_MAINTENANCE", "false"):
            summary = await run_queue_ingestion()
            assert summary["success"] == 1
            
        # Verify document is completely removed
        assert service.get_max_document_version(doc_id, collection_name=col) == 0
        assert not service.verify_version_exists(doc_id, 2, collection_name=col)

    finally:
        if os.path.exists(temp_filepath):
            try: os.unlink(temp_filepath)
            except Exception: pass
