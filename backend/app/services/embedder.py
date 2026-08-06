import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
from app.config import settings

# Route SentenceTransformers and FastEmbed downloads to our local cache folder
os.environ["SENTENCE_TRANSFORMERS_HOME"] = settings.EMBEDDING_CACHE_DIR
os.environ["FASTEMBED_CACHE_PATH"] = settings.EMBEDDING_CACHE_DIR

import torch

class EmbedderService:
    def __init__(self):
        # 1. Initialize local HuggingFaceEmbeddings (Dense vectorizer)
        # Using BAAI/bge-small-en-v1.5 (dimension 384)
        model_name = "BAAI/bge-small-en-v1.5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 2. Initialize FastEmbedSparse (Sparse vectorizer for keyword search)
        # Using SPLADE: prithivida/Splade_PP_en_v1
        self.sparse_embeddings = FastEmbedSparse(
            model_name="prithivida/Splade_PP_en_v1",
            cache_dir=settings.EMBEDDING_CACHE_DIR
        )

# Initialize a singleton instance
embedder_manager = EmbedderService()
