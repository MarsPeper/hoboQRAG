import pytest
from unittest.mock import MagicMock, patch
import torch
from app.services.embedder import EmbedderService

def test_embedder_initialization():
    """Test that both dense and sparse embedders initialize correctly."""
    service = EmbedderService()
    assert service.dense_embeddings is not None
    assert service.sparse_embeddings is not None

def test_embedding_generation():
    """Test dense and sparse embedding dimensions and output format."""
    service = EmbedderService()
    dense_vec = service.dense_embeddings.embed_query("test query")
    assert len(dense_vec) == 384
    
    sparse_vec = service.sparse_embeddings.embed_query("test query")
    assert hasattr(sparse_vec, "indices")
    assert hasattr(sparse_vec, "values")

def test_batch_embedding():
    """Test embedding multiple documents in a batch."""
    service = EmbedderService()
    texts = ["hello", "world"]
    dense_vecs = service.dense_embeddings.embed_documents(texts)
    assert len(dense_vecs) == 2
    for vec in dense_vecs:
        assert len(vec) == 384

def test_empty_input_handling():
    """Test embedding an empty string."""
    service = EmbedderService()
    dense_vec = service.dense_embeddings.embed_query("")
    assert len(dense_vec) == 384

@patch("torch.cuda.is_available")
def test_cuda_detection_and_fallback(mock_cuda):
    """Test that device routes to CUDA if available, else falls back to CPU."""
    # Scenario 1: CUDA is available
    mock_cuda.return_value = True
    with patch("app.services.embedder.HuggingFaceEmbeddings") as mock_hf:
        service = EmbedderService()
        mock_hf.assert_called_once()
        kwargs = mock_hf.call_args[1]
        assert kwargs["model_kwargs"]["device"] == "cuda"
        
    # Scenario 2: CUDA is unavailable (CPU fallback)
    mock_cuda.return_value = False
    with patch("app.services.embedder.HuggingFaceEmbeddings") as mock_hf:
        service = EmbedderService()
        mock_hf.assert_called_once()
        kwargs = mock_hf.call_args[1]
        assert kwargs["model_kwargs"]["device"] == "cpu"
