import logging
from typing import Generator
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.models.schemas import ChatRequest, DocumentListResponse, CreateCollectionRequest
from app.core.parser import DocumentParser, LangChainSplitter
from app.services.qdrant import qdrant_service
from app.services.llm_service import llm_service

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hoboQRAG")

# 2. Initialize FastAPI Application
app = FastAPI(
    title="hoboQRAG API (LangChain Edition)",
    description="Completely local production-grade RAG pipeline serving Tech Support using LangChain.",
    version="1.0.0"
)

# 3. Configure CORS (Allows WPF client connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Initialize Prometheus Instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# =====================================================================
# Endpoints
# =====================================================================

@app.post("/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (PDF, Text, Markdown), parses it into LangChain Documents,
    chunks it using RecursiveCharacterTextSplitter, embeds and saves it in Qdrant.
    """
    logger.info(f"Received file upload request: {file.filename}")
    
    # Save file temporarily
    temp_path = settings.UPLOAD_DIR / file.filename
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Parse text content into a list of LangChain Document objects
        raw_documents = DocumentParser.load_file(temp_path)
        
        # Split documents using LangChain RecursiveCharacterTextSplitter
        splitter = LangChainSplitter()
        chunks = splitter.split_documents(raw_documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document contains no readable text.")

        # Upload documents to Qdrant via QdrantVectorStore
        qdrant_service.insert_documents(chunks)
        
        return {
            "message": f"Successfully parsed and indexed file '{file.filename}'.",
            "chunk_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"Failed to process upload for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    Handles similarity search (Dense + Sparse/BM25) over Qdrant,
    reranks results using FlashrankRerank, and streams tokens from vLLM.
    """
    logger.info(f"Received chat request prompt: '{request.prompt}'")
    
    try:
        # 1. Retrieve initial candidate chunks (fetching 15 documents)
        raw_candidates = qdrant_service.search_hybrid(request.prompt, limit=15)
        
        # 2. Local Reranking (compress down to request.top_k, e.g. 4)
        reranked_docs = qdrant_service.rerank(
            query=request.prompt, 
            documents=raw_candidates, 
            top_n=request.top_k
        )
        
        # 3. Construct Context and Citations
        context_blocks = []
        citations = []
        
        for idx, doc in enumerate(reranked_docs):
            meta = doc.metadata
            file_name = meta.get("file_name", "Unknown File")
            page = meta.get("page", 1)
            
            # Format block text for LLM injection
            block_text = f"[Doc {idx+1}] File: {file_name} (Page {page})\nContent:\n{doc.page_content}"
            context_blocks.append(block_text)
            
            # Keep citation info
            score = meta.get("relevance_score")
            if score is None:
                score = meta.get("base_score", 0.0)

            citations.append({
                "index": idx + 1,
                "file_name": file_name,
                "page": page,
                "score": float(score)
            })
            
        context_str = "\n\n".join(context_blocks)
        
        # Log retrieved sources
        logger.info(f"Retrieved {len(citations)} source blocks for prompt. Top source: {citations[0] if citations else 'None'}")
        
        # 4. Stream response
        async def response_generator():
            # Translate message history schema
            history_list = []
            if request.history:
                history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
                
            # Stream actual LLM tokens via LangChain ChatOpenAI
            async for token in llm_service.stream_chat(
                prompt=request.prompt,
                context=context_str,
                history=history_list
            ):
                yield token

        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Failed to process chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal chat error: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Returns a list of all unique documents currently stored in Qdrant with chunk counts."""
    logger.info("Received request to list indexed documents.")
    try:
        files = qdrant_service.list_unique_files()
        return {"documents": files}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/documents")
async def delete_document(file_name: str = Query(..., description="The name of the file to delete")):
    """Deletes all vector points associated with the specified filename."""
    logger.info(f"Received request to delete document: '{file_name}'")
    success = qdrant_service.delete_file(file_name)
    if success:
        return {"message": f"Successfully deleted document '{file_name}' from the vector database."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete document '{file_name}'.")


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/collections")
async def list_collections():
    """Lists all collections in Qdrant."""
    try:
        collections = qdrant_service.client.get_collections().collections
        return {"collections": [c.name for c in collections]}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.post("/api/collections")
async def create_collection(req: CreateCollectionRequest):
    """Creates a new collection in Qdrant."""
    try:
        # Check duplicate
        if qdrant_service.collection_exists(req.name):
            raise HTTPException(status_code=400, detail=f"Collection '{req.name}' already exists.")
        success = qdrant_service.create_collection(req.name)
        if success:
            return {"message": f"Collection '{req.name}' created successfully."}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to create collection '{req.name}'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@app.post("/api/collections/{collection_name}/documents", status_code=201)
async def upload_document_to_collection(collection_name: str, file: UploadFile = File(...)):
    """
    Uploads a document (PDF, Text, Markdown), parses it into LangChain Documents,
    chunks it, embeds and saves it in a specific Qdrant collection.
    """
    logger.info(f"Received file upload request for collection '{collection_name}': {file.filename}")
    
    # Check if collection exists
    try:
        if not qdrant_service.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant connection error: {str(e)}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")

    # Save file temporarily
    temp_path = settings.UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file upload.")
            
        with open(temp_path, "wb") as f:
            f.write(content)
            
        # Parse text content into a list of LangChain Document objects
        try:
            raw_documents = DocumentParser.load_file(temp_path)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        # Split documents using LangChain RecursiveCharacterTextSplitter
        splitter = LangChainSplitter()
        chunks = splitter.split_documents(raw_documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document contains no readable text.")

        # Upload documents to Qdrant via QdrantVectorStore
        qdrant_service.insert_documents(chunks, collection_name=collection_name)
        
        return {
            "message": f"Successfully parsed and indexed file '{file.filename}' into collection '{collection_name}'.",
            "chunk_count": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process upload for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/api/chat")
async def chat_with_collection(request: ChatRequest):
    """
    Handles similarity search (Dense + Sparse/BM25) over a specific Qdrant collection,
    reranks results using FlashrankRerank, and streams tokens from vLLM.
    """
    col = request.collection_name or settings.QDRANT_COLLECTION_NAME
    logger.info(f"Received chat request prompt: '{request.prompt}' for collection '{col}'")
    
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Question/prompt cannot be empty.")

    try:
        if not qdrant_service.collection_exists(col):
            raise HTTPException(status_code=404, detail=f"Collection '{col}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant unavailable or connection failed: {str(e)}")

    try:
        # 1. Retrieve initial candidate chunks (fetching 15 documents)
        raw_candidates = qdrant_service.search_hybrid(request.prompt, limit=15, collection_name=col)
        
        # 2. Local Reranking (compress down to request.top_k, e.g. 4)
        reranked_docs = qdrant_service.rerank(
            query=request.prompt, 
            documents=raw_candidates, 
            top_n=request.top_k
        )
        
        # 3. Construct Context and Citations
        context_blocks = []
        citations = []
        
        for idx, doc in enumerate(reranked_docs):
            meta = doc.metadata
            file_name = meta.get("file_name", "Unknown File")
            page = meta.get("page", 1)
            
            # Format block text for LLM injection
            block_text = f"[Doc {idx+1}] File: {file_name} (Page {page})\nContent:\n{doc.page_content}"
            context_blocks.append(block_text)
            
            # Keep citation info
            score = meta.get("relevance_score")
            if score is None:
                score = meta.get("base_score", 0.0)

            citations.append({
                "index": idx + 1,
                "file_name": file_name,
                "page": page,
                "score": float(score)
            })
            
        context_str = "\n\n".join(context_blocks)
        
        # Log retrieved sources
        logger.info(f"Retrieved {len(citations)} source blocks for prompt. Top source: {citations[0] if citations else 'None'}")
        
        # 4. Stream response
        async def response_generator():
            # Translate message history schema
            history_list = []
            if request.history:
                history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
                
            # Stream actual LLM tokens via LangChain ChatOpenAI
            async for token in llm_service.stream_chat(
                prompt=request.prompt,
                context=context_str,
                history=history_list
            ):
                yield token

        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Failed to process chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal chat error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("hoboQRAG FastAPI backend (LangChain Edition) successfully started.")
    logger.info(f"FastAPI host: {settings.HOST}:{settings.PORT}")

