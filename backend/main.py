import logging
import os
import uuid
from typing import Generator
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
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
            from app.services.chat_gate import chat_gate
            # Translate message history schema
            history_list = []
            if request.history:
                history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
                
            # Stream actual LLM tokens via LangChain ChatOpenAI
            await chat_gate.enter_chat()
            try:
                async for token in llm_service.stream_chat(
                    prompt=request.prompt,
                    context=context_str,
                    history=history_list
                ):
                    yield token
            finally:
                await chat_gate.exit_chat()

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
            from app.services.chat_gate import chat_gate
            # Translate message history schema
            history_list = []
            if request.history:
                history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
                
            # Stream actual LLM tokens via LangChain ChatOpenAI
            await chat_gate.enter_chat()
            try:
                async for token in llm_service.stream_chat(
                    prompt=request.prompt,
                    context=context_str,
                    history=history_list
                ):
                    yield token
            finally:
                await chat_gate.exit_chat()

        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Failed to process chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal chat error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the premium minimalist Document Ingestion Queue management dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>hoboQRAG Ingestion Dashboard</title>
    <style>
        :root {
            --bg-color: #0b0c10;
            --card-bg: #1f2833;
            --primary: #66fcf1;
            --primary-hover: #45a29e;
            --text-main: #c5c6c7;
            --text-bright: #ffffff;
            --border: #45a29e;
            --danger: #ff4d4d;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: system-ui, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
        }
        h1 {
            color: var(--text-bright);
            font-weight: 300;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 30px;
            letter-spacing: 1px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background-color: var(--card-bg);
            border: 1px solid rgba(102, 252, 241, 0.15);
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }
        .stat-val {
            font-size: 1.8rem;
            color: var(--primary);
            font-weight: 600;
            margin-top: 5px;
        }
        .stat-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.7;
        }
        .card {
            background-color: var(--card-bg);
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        h2 {
            color: var(--text-bright);
            font-size: 1.2rem;
            margin-top: 0;
            font-weight: 500;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        th, td {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        th {
            color: var(--text-bright);
            font-weight: 500;
        }
        .btn {
            background-color: var(--primary);
            color: #0b0c10;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background-color: var(--primary-hover);
        }
        .btn-danger {
            background-color: var(--danger);
            color: white;
            padding: 6px 12px;
            font-size: 0.8rem;
        }
        .btn-secondary {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--primary);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 0.85rem;
            color: var(--text-bright);
        }
        select, input[type="text"], input[type="file"] {
            width: 100%;
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 10px;
            border-radius: 4px;
            color: var(--text-bright);
            box-sizing: border-box;
        }
        select:focus, input[type="text"]:focus {
            border-color: var(--primary);
            outline: none;
        }
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .badge-add { background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; }
        .badge-replace { background-color: rgba(241, 196, 15, 0.2); color: #f1c40f; }
        .badge-delete { background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; }
        
        .progress-box {
            background-color: rgba(102, 252, 241, 0.05);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 0.95rem;
            display: none;
        }
    </style>
</head>
<body>
    <h1>hoboQRAG Ingestion Dashboard</h1>
    
    <div id="progress-container" class="progress-box">
        <strong>Ingestion Active:</strong> Processing operation <span id="current-index">0</span>/<span id="total-index">0</span> (<span id="current-file">-</span>)
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Pending Operations</div>
            <div class="stat-val" id="stat-pending">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Next Scheduled Run</div>
            <div class="stat-val" id="stat-next" style="font-size: 1.1rem; padding-top: 10px;">00:00</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Ingest Status</div>
            <div class="stat-val" id="stat-status" style="font-size: 1.3rem; padding-top: 7px;">Idle</div>
        </div>
    </div>

    <div class="card">
        <h2>Queue a Document Operation</h2>
        <form id="enqueue-form" enctype="multipart/form-data">
            <div class="form-group">
                <label for="collection">Collection Target</label>
                <select id="collection" name="collection" required>
                    <option value="tech_support_kb">tech_support_kb</option>
                </select>
            </div>
            <div class="form-group">
                <label for="operation">Action</label>
                <select id="operation" name="operation" onchange="adjustFormFields()" required>
                    <option value="ADD">ADD - Upload new document</option>
                    <option value="REPLACE">REPLACE - Overwrite existing document</option>
                    <option value="DELETE">DELETE - Remove existing document</option>
                </select>
            </div>
            
            <div class="form-group" id="doc-id-container" style="display:none;">
                <label for="document_id">Target Document ID</label>
                <select id="document_id" name="document_id">
                    <option value="">Select a document...</option>
                </select>
            </div>

            <div class="form-group" id="file-container">
                <label for="file">Document File</label>
                <input type="file" id="file" name="file">
            </div>

            <button type="submit" class="btn">Add to Ingestion Queue</button>
        </form>
    </div>

    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
            <h2 style="border: none; margin: 0; padding: 0;">Pending Queue</h2>
            <div>
                <button onclick="processQueueNow()" class="btn btn-secondary" style="padding: 6px 12px; font-size: 0.8rem;">Process Queue Now</button>
            </div>
        </div>
        <table id="queue-table">
            <thead>
                <tr>
                    <th>Document</th>
                    <th>Action</th>
                    <th>Queued By</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="4" style="text-align: center; opacity: 0.5;">No pending operations.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Job Execution History</h2>
        <table id="history-table">
            <thead>
                <tr>
                    <th>Job Started</th>
                    <th>Status</th>
                    <th>Ops</th>
                    <th>Success</th>
                    <th>Failed</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="6" style="text-align: center; opacity: 0.5;">No past runs found.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <script>
        async function fetchCollections() {
            try {
                const res = await fetch('/api/collections');
                const data = await res.json();
                const sel = document.getElementById('collection');
                sel.innerHTML = '';
                data.collections.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    sel.appendChild(opt);
                });
            } catch (e) { console.error("Error loading collections:", e); }
        }

        async function fetchDocuments() {
            try {
                const res = await fetch(`/documents`);
                const data = await res.json();
                const sel = document.getElementById('document_id');
                sel.innerHTML = '<option value="">Select a document...</option>';
                if(data.documents) {
                    data.documents.forEach(doc => {
                        const opt = document.createElement('option');
                        opt.value = doc.file_name;
                        opt.textContent = `${doc.file_name} (${doc.chunk_count} chunks)`;
                        sel.appendChild(opt);
                    });
                }
            } catch (e) { console.error("Error loading documents:", e); }
        }

        function adjustFormFields() {
            const op = document.getElementById('operation').value;
            const docContainer = document.getElementById('doc-id-container');
            const fileContainer = document.getElementById('file-container');
            
            if (op === 'ADD') {
                docContainer.style.display = 'none';
                fileContainer.style.display = 'block';
                document.getElementById('document_id').required = false;
                document.getElementById('file').required = true;
            } else if (op === 'REPLACE') {
                docContainer.style.display = 'block';
                fileContainer.style.display = 'block';
                document.getElementById('document_id').required = true;
                document.getElementById('file').required = true;
                fetchDocuments();
            } else if (op === 'DELETE') {
                docContainer.style.display = 'block';
                fileContainer.style.display = 'none';
                document.getElementById('document_id').required = true;
                document.getElementById('file').required = false;
                fetchDocuments();
            }
        }

        async function loadQueue() {
            try {
                const res = await fetch('/api/ingestion/queue');
                const queue = await res.json();
                const tbody = document.querySelector('#queue-table tbody');
                tbody.innerHTML = '';
                
                document.getElementById('stat-pending').textContent = queue.length;
                
                if (queue.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; opacity: 0.5;">No pending operations.</td></tr>';
                    return;
                }
                
                queue.forEach(item => {
                    const tr = document.createElement('tr');
                    const badgeClass = `badge badge-${item.operation.toLowerCase()}`;
                    tr.innerHTML = `
                        <td>${item.document_id} ${item.filename ? '<br><small style="opacity:0.6;">File: ' + item.filename + '</small>' : ''}</td>
                        <td><span class="${badgeClass}">${item.operation}</span></td>
                        <td>${item.queued_by}</td>
                        <td><button onclick="cancelItem('${item.id}')" class="btn btn-danger">Cancel</button></td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (e) { console.error(e); }
        }

        async function loadStatus() {
            try {
                const res = await fetch('/api/ingestion/status');
                const status = await res.json();
                
                const nextRun = new Date(status.next_scheduled_run);
                document.getElementById('stat-next').textContent = nextRun.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                const statStatus = document.getElementById('stat-status');
                const progContainer = document.getElementById('progress-container');
                
                if (status.is_processing) {
                    statStatus.textContent = 'Syncing';
                    statStatus.style.color = 'var(--primary)';
                    progContainer.style.display = 'block';
                    document.getElementById('current-index').textContent = status.current_operation_index;
                    document.getElementById('total-index').textContent = status.total_operations;
                    document.getElementById('current-file').textContent = status.current_operation_filename;
                } else {
                    statStatus.textContent = 'Idle';
                    statStatus.style.color = 'var(--text-main)';
                    progContainer.style.display = 'none';
                }
            } catch (e) { console.error(e); }
        }

        async function loadHistory() {
            try {
                const res = await fetch('/api/ingestion/history');
                const history = await res.json();
                const tbody = document.querySelector('#history-table tbody');
                tbody.innerHTML = '';
                
                if (history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; opacity: 0.5;">No past runs found.</td></tr>';
                    return;
                }
                
                history.forEach(run => {
                    const tr = document.createElement('tr');
                    const dateStr = new Date(run.started_at).toLocaleString();
                    const statusColor = run.status === 'completed' ? '#2ecc71' : '#e74c3c';
                    const summary = run.summary || {};
                    
                    tr.innerHTML = `
                        <td>${dateStr}</td>
                        <td style="color:${statusColor}; font-weight:600;">${run.status.toUpperCase()}</td>
                        <td>${summary.total_processed || 0}</td>
                        <td>${summary.success || 0}</td>
                        <td>${summary.failed || 0}</td>
                        <td>${summary.duration_seconds || 0}s</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (e) { console.error(e); }
        }

        async function cancelItem(id) {
            if(!confirm("Are you sure you want to cancel this pending change?")) return;
            try {
                const res = await fetch(`/api/ingestion/queue/${id}`, { method: 'DELETE' });
                const val = await res.json();
                alert(val.message);
                loadQueue();
            } catch (e) { alert(e); }
        }

        async function processQueueNow() {
            try {
                const res = await fetch('/api/ingestion/process', { method: 'POST' });
                const val = await res.json();
                alert(val.message);
                loadStatus();
            } catch (e) { alert(e); }
        }

        document.getElementById('enqueue-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData();
            
            const op = document.getElementById('operation').value;
            const collection = document.getElementById('collection').value;
            let docId = document.getElementById('document_id').value;
            
            formData.append('collection', collection);
            formData.append('operation', op);
            formData.append('queued_by', 'Support Web UI');
            
            if (op === 'ADD') {
                const fileInput = document.getElementById('file');
                if(!fileInput.files.length) return alert("Select a file.");
                const file = fileInput.files[0];
                formData.append('file', file);
                formData.append('document_id', file.name);
            } else if (op === 'REPLACE') {
                const fileInput = document.getElementById('file');
                if(!fileInput.files.length) return alert("Select a replacement file.");
                formData.append('file', fileInput.files[0]);
                formData.append('document_id', docId);
            } else if (op === 'DELETE') {
                formData.append('document_id', docId);
            }
            
            try {
                const res = await fetch('/api/ingestion/queue', {
                    method: 'POST',
                    body: formData
                });
                const val = await res.json();
                if(res.status >= 400) {
                    alert(`Error: ${val.detail}`);
                } else {
                    alert(val.message);
                    form.reset();
                    adjustFormFields();
                    loadQueue();
                }
            } catch (err) {
                alert(err);
            }
        });

        // Init
        document.addEventListener('DOMContentLoaded', async () => {
            await fetchCollections();
            adjustFormFields();
            loadQueue();
            loadStatus();
            loadHistory();
            
            setInterval(() => {
                loadQueue();
                loadStatus();
                loadHistory();
            }, 5000);
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/api/ingestion/queue")
async def get_ingestion_queue():
    """Retrieves pending enqueued operations."""
    from app.services.queue_db import queue_db
    return queue_db.get_queue()


@app.post("/api/ingestion/queue", status_code=201)
async def enqueue_operation(
    collection: str = Form(...),
    document_id: str = Form(...),
    operation: str = Form(...),
    queued_by: str = Form("user"),
    file: UploadFile = File(None)
):
    """Enqueues a new operation to the ingestion queue (ADD, DELETE, REPLACE)."""
    operation = operation.upper()
    if operation not in ["ADD", "DELETE", "REPLACE"]:
        raise HTTPException(status_code=400, detail=f"Invalid operation '{operation}'. Must be ADD, DELETE, or REPLACE.")

    from app.services.qdrant import qdrant_service
    if not qdrant_service.collection_exists(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' does not exist.")

    if operation in ["DELETE", "REPLACE"]:
        files = qdrant_service.list_unique_files(collection_name=collection)
        existing_doc_ids = [f["file_name"] for f in files]
        if document_id not in existing_doc_ids:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' does not exist in collection '{collection}'.")

    filename = ""
    filepath = ""

    if operation in ["ADD", "REPLACE"]:
        if not file:
            raise HTTPException(status_code=400, detail=f"File upload is required for {operation} operation.")
        
        filename = file.filename
        op_id = str(uuid.uuid4())
        filepath = str(settings.QUEUE_FILES_DIR / f"{op_id}_{filename}")
        
        try:
            with open(filepath, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write uploaded file payload: {str(e)}")

    from app.services.queue_db import queue_db
    try:
        result = queue_db.resolve_conflicts_and_enqueue(
            collection=collection,
            document_id=document_id,
            operation=operation,
            filename=filename,
            filepath=filepath,
            queued_by=queued_by
        )
        
        if result == "CANCELLED_PENDING_ADD":
            return {"message": f"Successfully cancelled pending ADD for document '{document_id}' (net-zero change).", "operation_id": None}
        
        return {"message": f"Successfully enqueued {operation} operation.", "operation_id": result}
    except ValueError as e:
        if filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except Exception:
                pass
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        if filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Database error enqueuing operation: {str(e)}")


@app.delete("/api/ingestion/queue/{operation_id}")
async def cancel_queue_operation(operation_id: str):
    """Cancels a pending operation from the queue."""
    from app.services.queue_db import queue_db
    item = queue_db.get_queue_item(operation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Queue operation not found.")
    
    if item["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot cancel operation with status '{item['status']}'.")
    
    try:
        queue_db.delete_queue_item(operation_id)
        if item["filepath"] and os.path.exists(item["filepath"]):
            try:
                os.unlink(item["filepath"])
            except Exception:
                pass
        return {"message": f"Operation {operation_id} successfully cancelled."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingestion/process")
async def trigger_ingestion_processing():
    """Manually triggers processing of the queued operations."""
    from app.services.sync import run_queue_ingestion, ingestion_lock
    import asyncio
    
    if ingestion_lock.locked():
        raise HTTPException(status_code=409, detail="An ingestion run is already in progress.")
        
    asyncio.create_task(run_queue_ingestion())
    return {"message": "Ingestion job successfully triggered in background."}


@app.get("/api/ingestion/status")
async def get_ingestion_status():
    """Retrieves current status of the ingestion worker and queue stats."""
    from app.services.sync import ingestion_status, ingestion_lock
    from app.services.queue_db import queue_db
    from datetime import time as dt_time, timedelta
    
    try:
        h, m = map(int, settings.INGESTION_SCHEDULE.split(":"))
    except Exception:
        h, m = 0, 0
    now = datetime.now()
    target = datetime.combine(now.date(), dt_time(h, m))
    if target <= now:
        target = target + timedelta(days=1)
    
    last_success = None
    last_failure = None
    
    all_history = queue_db.get_history(limit=20)
    for run in all_history:
        if run["status"] == "completed" and last_success is None:
            last_success = run["completed_at"]
        elif run["status"] == "failed" and last_failure is None:
            last_failure = run["completed_at"]

    pending_ops = len(queue_db.get_queue(status="pending"))

    return {
        "is_processing": ingestion_status.is_processing,
        "lock_active": ingestion_lock.locked(),
        "total_operations": ingestion_status.total_ops,
        "current_operation_index": ingestion_status.current_op_index,
        "current_operation_filename": ingestion_status.current_op_filename,
        "pending_queue_size": pending_ops,
        "next_scheduled_run": target.isoformat(),
        "last_successful_run": last_success,
        "last_failed_run": last_failure,
        "last_run_summary": ingestion_status.last_run_summary
    }


@app.get("/api/ingestion/history")
async def get_ingestion_history(limit: int = 50):
    """Retrieves list of past job execution runs."""
    from app.services.queue_db import queue_db
    return queue_db.get_history(limit=limit)


@app.on_event("startup")
async def startup_event():
    import asyncio
    from app.services.sync import scheduled_sync_loop
    logger.info("hoboQRAG FastAPI backend (LangChain Edition) successfully started.")
    logger.info(f"FastAPI host: {settings.HOST}:{settings.PORT}")
    asyncio.create_task(scheduled_sync_loop())

