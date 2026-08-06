import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.services.qdrant import qdrant_service

def test_reranker_empty_list():
    """Test that reranking an empty candidate list returns empty list."""
    res = qdrant_service.rerank("query", [])
    assert res == []

def test_reranker_top_k():
    """Test that the reranker returns exactly top_n elements."""
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3")
    ]
    res = qdrant_service.rerank("query", docs, top_n=2)
    assert len(res) == 2

def test_reranker_ordering():
    """Test that reranker scores and orders candidates properly."""
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2")
    ]
    res = qdrant_service.rerank("query", docs)
    assert len(res) == 2
    assert res[0].metadata["relevance_score"] == 0.9
    assert res[1].metadata["relevance_score"] == 0.8

def test_reranker_failure_fallback():
    """Test fallback when the reranking compressor raises an exception."""
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3")
    ]
    
    # Force FlashrankRerank initialization to throw RuntimeError
    with patch("langchain_community.document_compressors.FlashrankRerank", side_effect=RuntimeError("ONNX Runtime error")):
        res = qdrant_service.rerank("query", docs, top_n=2)
        # Should gracefully fallback to first top_n items
        assert len(res) == 2
        assert res[0].page_content == "doc1"
        assert res[1].page_content == "doc2"
