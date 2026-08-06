import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.core.parser import DocumentParser, LangChainSplitter

def test_parse_text_normal(tmp_path):
    """Test text parsing with standard text document."""
    file_path = tmp_path / "normal.txt"
    file_path.write_text("Hello world! This is a test document.", encoding="utf-8")
    
    docs = DocumentParser.load_file(file_path)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello world! This is a test document."
    assert docs[0].metadata["file_name"] == "normal.txt"
    assert docs[0].metadata["page"] == 1

def test_parse_text_empty(tmp_path):
    """Test parsing an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    
    docs = DocumentParser.load_file(file_path)
    assert len(docs) == 1
    assert docs[0].page_content == ""

def test_parse_text_unicode(tmp_path):
    """Test parsing document containing unusual unicode characters."""
    file_path = tmp_path / "unicode.txt"
    file_path.write_text("Unicode test: 🤠 🚀 日本語. Splendid!", encoding="utf-8")
    
    docs = DocumentParser.load_file(file_path)
    assert len(docs) == 1
    assert "🤠 🚀 日本語" in docs[0].page_content

def test_parse_unsupported(tmp_path):
    """Test exception for unsupported file type."""
    file_path = tmp_path / "unsupported.png"
    file_path.write_text("binary-data", encoding="utf-8")
    
    with pytest.raises(ValueError) as excinfo:
        DocumentParser.load_file(file_path)
    assert "Unsupported file extension" in str(excinfo.value)

@patch("app.core.parser.PdfReader")
def test_parse_pdf_normal(mock_pdf_reader, tmp_path):
    """Test PDF parsing by mocking PdfReader."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page 1 context data."
    
    mock_reader_instance = mock_pdf_reader.return_value
    mock_reader_instance.pages = [mock_page]
    
    file_path = tmp_path / "test.pdf"
    file_path.write_text("fake pdf bytes", encoding="utf-8")
    
    docs = DocumentParser.load_file(file_path)
    assert len(docs) == 1
    assert docs[0].page_content == "Page 1 context data."
    assert docs[0].metadata["file_name"] == "test.pdf"
    assert docs[0].metadata["page"] == 1

def test_splitter_chunking():
    """Test chunking parameters, chunk sizes, overlap and metadata injection."""
    docs = [
        Document(
            page_content="Word " * 200,  # ~1000 characters
            metadata={"file_name": "test.txt", "page": 1}
        )
    ]
    splitter = LangChainSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)
        assert len(chunk.page_content) <= 120  # Allow some wiggle room for separators
