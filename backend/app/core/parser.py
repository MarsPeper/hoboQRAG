import re
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: Path) -> List[Document]:
        """Extracts text from a PDF file and returns a list of LangChain Documents (one per page)."""
        reader = PdfReader(file_path)
        documents = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Store the page content and metadata
                documents.append(
                    Document(
                        page_content=page_text.strip(),
                        metadata={
                            "file_name": file_path.name,
                            "page": i + 1
                        }
                    )
                )
        return documents

    @staticmethod
    def parse_text(file_path: Path) -> List[Document]:
        """Extracts text from a plain text or markdown file and returns as a LangChain Document."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [
            Document(
                page_content=content.strip(),
                metadata={
                    "file_name": file_path.name,
                    "page": 1
                }
            )
        ]

    @classmethod
    def load_file(cls, file_path: Path) -> List[Document]:
        """Determines file type and extracts as a list of LangChain Documents."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".txt", ".md", ".json", ".log"]:
            return cls.parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


class LangChainSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # Splitting by paragraphs, sentences, words, etc. dynamically
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of LangChain Documents into chunked LangChain Documents."""
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk index metadata for tracing
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
        return chunks
