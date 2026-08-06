import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue
)
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.document_compressors import FlashrankRerank
from app.config import settings
from app.services.embedder import embedder_manager

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        # 1. Initialize standalone Qdrant client
        logger.info(f"Connecting to Qdrant server at: {settings.QDRANT_URL}")
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # 2. Ensure collection configuration (Dense + Sparse)
        self._ensure_collection()

        # 3. Create LangChain QdrantVectorStore wrapper
        # Set RetrievalMode to HYBRID (combining Dense + Sparse/BM25)
        logger.info(f"Initializing LangChain QdrantVectorStore with HYBRID retrieval")
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedder_manager.dense_embeddings,
            sparse_embedding=embedder_manager.sparse_embeddings,
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Failed to check collections in Qdrant: {e}")
            raise e

    def create_collection(self, collection_name: str) -> bool:
        """Creates a collection with both dense and sparse configurations. Returns True if created, False if already exists."""
        if self.collection_exists(collection_name):
            return False

        try:
            dense_dim = len(embedder_manager.dense_embeddings.embed_query("test"))
        except Exception as e:
            logger.warning(f"Could not automatically detect embedding dimension: {e}. Defaulting to 384.")
            dense_dim = 384

        logger.info(f"Creating collection '{collection_name}' with dense_dim={dense_dim} and sparse support...")
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dense_dim,
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)
                )
            }
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
        return True

    def _ensure_collection(self):
        """Creates the collection with both dense and sparse configurations if it doesn't exist."""
        try:
            self.create_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to ensure default collection: {e}")

    def get_vector_store(self, collection_name: str = None) -> QdrantVectorStore:
        """Gets a QdrantVectorStore wrapper for a given collection."""
        col = collection_name or self.collection_name
        return QdrantVectorStore(
            client=self.client,
            collection_name=col,
            embedding=embedder_manager.dense_embeddings,
            sparse_embedding=embedder_manager.sparse_embeddings,
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID
        )

    def insert_documents(self, documents: List[Document], collection_name: str = None):
        """Uploads a list of LangChain Document objects to Qdrant."""
        if not documents:
            return
        col = collection_name or self.collection_name
        logger.info(f"Uploading {len(documents)} split documents to Qdrant collection '{col}' via LangChain...")
        store = self.get_vector_store(col)
        store.add_documents(documents)
        logger.info("LangChain document upload complete.")

    def search_hybrid(self, query: str, limit: int = 15, collection_name: str = None) -> List[Document]:
        """
        Executes a hybrid search (Dense + Sparse) in Qdrant.
        Unpacks scores and stores them in document metadata.
        """
        col = collection_name or self.collection_name
        logger.info(f"Performing LangChain hybrid similarity search for: '{query}' in collection '{col}'")
        store = self.get_vector_store(col)
        docs_with_scores = store.similarity_search_with_score(query, k=limit)
        
        results = []
        for doc, score in docs_with_scores:
            doc.metadata["base_score"] = float(score)
            results.append(doc)
            
        # Multi-version resolution: Filter out older document versions
        max_versions = {}
        for doc in results:
            doc_id = doc.metadata.get("document_id")
            ver = doc.metadata.get("document_version")
            if doc_id and ver is not None:
                try:
                    max_versions[doc_id] = max(max_versions.get(doc_id, 0), int(ver))
                except (ValueError, TypeError):
                    pass

        filtered_results = []
        for doc in results:
            doc_id = doc.metadata.get("document_id")
            ver = doc.metadata.get("document_version")
            if doc_id and ver is not None:
                try:
                    if int(ver) == max_versions[doc_id]:
                        filtered_results.append(doc)
                except (ValueError, TypeError):
                    filtered_results.append(doc)
            else:
                filtered_results.append(doc)
                
        return filtered_results

    def rerank(self, query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
        """
        Reranks retrieved candidate documents using FlashrankRerank.
        """
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} documents using FlashrankRerank...")
        try:
            compressor = FlashrankRerank(
                model="ms-marco-MiniLM-L-12-v2",
                top_n=top_n,
                cache_dir=settings.EMBEDDING_CACHE_DIR
            )
            return compressor.compress_documents(documents, query)
        except Exception as e:
            logger.error(f"FlashrankRerank failed: {e}. Falling back to top {top_n} base candidates.")
            return documents[:top_n]

    def list_unique_files(self, collection_name: str = None) -> List[Dict[str, Any]]:
        """Scrolls the collection using the Qdrant client to get unique uploaded filenames."""
        col = collection_name or self.collection_name
        files = {}
        limit = 100
        offset = None

        logger.info(f"Scrolling Qdrant collection '{col}' payloads to retrieve unique files...")
        while True:
            records, offset = self.client.scroll(
                collection_name=col,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for rec in records:
                if not rec.payload:
                    continue
                metadata = rec.payload.get("metadata")
                if not metadata or not isinstance(metadata, dict):
                    continue
                name = metadata.get("file_name")
                if not name:
                    continue
                
                if name not in files:
                    files[name] = {
                        "file_name": name,
                        "chunk_count": 1
                    }
                else:
                    files[name]["chunk_count"] += 1

            if offset is None:
                break

        return list(files.values())

    def delete_file(self, filename: str, collection_name: str = None) -> bool:
        """Deletes all vector points matching the filename payload."""
        col = collection_name or self.collection_name
        logger.info(f"Deleting Qdrant points with file_name: {filename} from collection '{col}'")
        try:
            result = self.client.delete(
                collection_name=col,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_name",
                            match=MatchValue(value=filename)
                        )
                    ]
                )
            )
            logger.info(f"Deleted points matching {filename}. Result: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {filename} from Qdrant: {e}")
            return False

    def get_max_document_version(self, document_id: str, collection_name: str = None) -> int:
        """Scrolls Qdrant to find the highest document_version stored for this document_id."""
        col = collection_name or self.collection_name
        try:
            records, _ = self.client.scroll(
                collection_name=col,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if records and records[0].payload:
                metadata = records[0].payload.get("metadata", {})
                ver = metadata.get("document_version")
                if ver is not None:
                    return int(ver)
            return 0
        except Exception as e:
            logger.error(f"Error fetching max version for document {document_id}: {e}")
            return 0

    def verify_version_exists(self, document_id: str, version: int, collection_name: str = None) -> bool:
        """Verifies that at least one point exists in Qdrant with this document_id and version."""
        col = collection_name or self.collection_name
        try:
            records, _ = self.client.scroll(
                collection_name=col,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)),
                        FieldCondition(key="metadata.document_version", match=MatchValue(value=version))
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            return len(records) > 0
        except Exception as e:
            logger.error(f"Verification failed for {document_id} version {version}: {e}")
            return False

    def delete_previous_versions(self, document_id: str, current_version: int, collection_name: str = None) -> bool:
        """Deletes all points for document_id where document_version is not equal to current_version."""
        col = collection_name or self.collection_name
        logger.info(f"Cleaning up previous versions for document_id '{document_id}' (keeping version: {current_version})")
        try:
            filter_query = Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="metadata.document_version",
                        match=MatchValue(value=current_version)
                    )
                ]
            )
            result = self.client.delete(
                collection_name=col,
                points_selector=filter_query
            )
            logger.info(f"Deleted old version points for '{document_id}'. Result: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete old versions for '{document_id}': {e}")
            return False


# Initialize a singleton instance
qdrant_service = QdrantService()
