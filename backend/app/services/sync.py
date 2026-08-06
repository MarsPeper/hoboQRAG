import os
import logging
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from typing import List, Dict, Any
import httpx
from prometheus_client import Counter, Gauge

from app.config import settings
from app.services.qdrant import qdrant_service
from app.services.queue_db import queue_db
from app.services.chat_gate import chat_gate
from app.core.parser import DocumentParser, LangChainSplitter

logger = logging.getLogger(__name__)

# Prometheus Ingestion Metrics
INGESTION_OPS_TOTAL = Counter(
    "ingestion_operations_total", 
    "Total number of ingestion operations processed",
    ["operation", "collection", "status"]
)
INGESTION_OPS_SUCCESS_TOTAL = Counter("ingestion_operations_success_total", "Total successful ingestion operations")
INGESTION_OPS_FAILED_TOTAL = Counter("ingestion_operations_failed_total", "Total failed ingestion operations")

INGESTION_DOCS_ADDED = Counter("ingestion_documents_added_total", "Total documents added")
INGESTION_DOCS_REPLACED = Counter("ingestion_documents_replaced_total", "Total documents replaced")
INGESTION_DOCS_DELETED = Counter("ingestion_documents_deleted_total", "Total documents deleted")

INGESTION_CHUNKS_CREATED = Counter("ingestion_chunks_created_total", "Total document chunks created")
INGESTION_DURATION = Gauge("ingestion_duration_seconds", "Duration of the last ingestion job run in seconds")
INGESTION_QUEUE_SIZE = Gauge("ingestion_queue_size", "Current number of pending operations in the queue")

INGESTION_LAST_SUCCESS = Gauge("ingestion_last_success_timestamp", "Epoch timestamp of the last successful ingestion run")
INGESTION_LAST_FAILURE = Gauge("ingestion_last_failure_timestamp", "Epoch timestamp of the last failed ingestion run")

# Global ingestion process lock
ingestion_lock = asyncio.Lock()

class IngestionStatus:
    def __init__(self):
        self.is_processing = False
        self.total_ops = 0
        self.current_op_index = 0
        self.current_op_filename = ""
        self.last_run_summary = None

ingestion_status = IngestionStatus()

class DockerManager:
    @staticmethod
    def stop_container(container_name: str) -> bool:
        try:
            import docker
            client = docker.from_env()
            container = client.containers.get(container_name)
            if container.status == "running":
                logger.info(f"Stopping container: {container_name}")
                container.stop(timeout=15)
                logger.info(f"Container {container_name} stopped successfully.")
                return True
            else:
                logger.info(f"Container {container_name} is not running (status: {container.status})")
                return True
        except Exception as e:
            logger.error(f"Could not stop container {container_name}: {e}. Proceeding.")
            return False

    @staticmethod
    def start_container(container_name: str) -> bool:
        try:
            import docker
            client = docker.from_env()
            container = client.containers.get(container_name)
            if container.status != "running":
                logger.info(f"Starting container: {container_name}")
                container.start()
                logger.info(f"Container {container_name} started successfully.")
                return True
            else:
                logger.info(f"Container {container_name} is already running.")
                return True
        except Exception as e:
            logger.error(f"Could not start container {container_name}: {e}. Proceeding.")
            return False


async def wait_for_vllm_ready(timeout: int = 120) -> bool:
    """Polls vLLM health models endpoint until it is online."""
    start_time = time.time()
    logger.info("Waiting for vLLM server to become online and healthy...")
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(settings.VLLM_URL + "/models", timeout=2.0)
                if response.status_code == 200:
                    logger.info("vLLM server is healthy and ready.")
                    return True
        except Exception:
            pass
        await asyncio.sleep(3)
    logger.warning("vLLM health check timed out.")
    return False


async def run_queue_ingestion() -> Dict[str, Any]:
    """
    Executes the ingestion worker flow:
    Acquire lock -> Load pending -> Stop vLLM (Maintenance) -> Process ops -> Verify -> Start vLLM -> Release
    """
    if ingestion_lock.locked():
        logger.warning("Ingestion job already running. Skipping execution.")
        return {"status": "skipped", "message": "An ingestion job is already in progress."}

    async with ingestion_lock:
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()
        
        # Load pending items
        pending_ops = queue_db.get_queue(status="pending")
        # Update queue size metric
        INGESTION_QUEUE_SIZE.set(len(pending_ops))

        if not pending_ops:
            logger.info("No pending ingestion operations found. Queue is empty.")
            return {"status": "empty", "message": "Queue is empty."}

        logger.info(f"Found {len(pending_ops)} pending operations in queue. Starting processing...")
        
        # Update status
        ingestion_status.is_processing = True
        ingestion_status.total_ops = len(pending_ops)
        ingestion_status.current_op_index = 0
        ingestion_status.current_op_filename = ""

        vllm_stopped = False
        gpu_maintenance = settings.INGESTION_GPU_MAINTENANCE == "true"
        
        # 1. Maintenance mode entry: Stop vLLM
        if gpu_maintenance:
            logger.info("GPU Maintenance Mode enabled. Draining chat requests...")
            chat_gate.block_new_chats()
            await chat_gate.wait_for_active_chats()
            
            logger.info("Stopping vLLM container to free GPU VRAM...")
            vllm_stopped = DockerManager.stop_container(settings.VLLM_CONTAINER_NAME)
            if vllm_stopped:
                await asyncio.sleep(5) # Let cooling settle

        success_count = 0
        failed_count = 0
        chunks_total = 0
        summary_ops = []

        try:
            for i, op in enumerate(pending_ops):
                ingestion_status.current_op_index = i + 1
                ingestion_status.current_op_filename = op["filename"]
                
                op_id = op["id"]
                col = op["collection"]
                doc_id = op["document_id"]
                action = op["operation"]
                filename = op["filename"]
                filepath = op["filepath"]
                
                logger.info(f"[{i+1}/{len(pending_ops)}] Processing {action} for {doc_id}")
                queue_db.update_queue_status(op_id, "processing", started_at=datetime.now().isoformat())
                
                op_success = False
                op_error = None
                chunks_count = 0
                
                try:
                    if action == "ADD":
                        # Ingest new document
                        if not filepath or not os.path.exists(filepath):
                            raise FileNotFoundError(f"Source file not found at path: {filepath}")
                        
                        max_ver = qdrant_service.get_max_document_version(doc_id, collection_name=col)
                        new_ver = max_ver + 1
                        
                        raw_docs = DocumentParser.load_file(Path(filepath))
                        splitter = LangChainSplitter()
                        chunks = splitter.split_documents(raw_docs)
                        
                        # Set version metadata
                        for chunk in chunks:
                            chunk.metadata["document_id"] = doc_id
                            chunk.metadata["document_version"] = new_ver
                            chunk.metadata["source"] = filename
                            
                        qdrant_service.insert_documents(chunks, collection_name=col)
                        
                        # Verify
                        if not qdrant_service.verify_version_exists(doc_id, new_ver, collection_name=col):
                            raise RuntimeError(f"Failed to verify version {new_ver} exists in Qdrant.")
                        
                        chunks_count = len(chunks)
                        op_success = True
                        INGESTION_DOCS_ADDED.inc()

                    elif action == "REPLACE":
                        # Ingest replacement
                        if not filepath or not os.path.exists(filepath):
                            raise FileNotFoundError(f"Source file not found at path: {filepath}")
                        
                        max_ver = qdrant_service.get_max_document_version(doc_id, collection_name=col)
                        new_ver = max_ver + 1
                        
                        raw_docs = DocumentParser.load_file(Path(filepath))
                        splitter = LangChainSplitter()
                        chunks = splitter.split_documents(raw_docs)
                        
                        # Set version metadata
                        for chunk in chunks:
                            chunk.metadata["document_id"] = doc_id
                            chunk.metadata["document_version"] = new_ver
                            chunk.metadata["source"] = filename
                            
                        qdrant_service.insert_documents(chunks, collection_name=col)
                        
                        # Verify new exists
                        if not qdrant_service.verify_version_exists(doc_id, new_ver, collection_name=col):
                            raise RuntimeError(f"Failed to verify replacement version {new_ver} exists in Qdrant.")
                        
                        # Delete old versions
                        qdrant_service.delete_previous_versions(doc_id, new_ver, collection_name=col)
                        
                        chunks_count = len(chunks)
                        op_success = True
                        INGESTION_DOCS_REPLACED.inc()

                    elif action == "DELETE":
                        # Delete document from Qdrant by setting version to 0 (which deletes all points)
                        qdrant_service.delete_previous_versions(doc_id, current_version=0, collection_name=col)
                        op_success = True
                        INGESTION_DOCS_DELETED.inc()

                    else:
                        raise ValueError(f"Unsupported queue operation: {action}")
                        
                except Exception as e:
                    op_error = str(e)
                    logger.error(f"Operation {op_id} failed: {e}", exc_info=True)

                # Finalize operation state
                if op_success:
                    queue_db.update_queue_status(op_id, "completed", completed_at=datetime.now().isoformat())
                    success_count += 1
                    chunks_total += chunks_count
                    INGESTION_OPS_SUCCESS_TOTAL.inc()
                    INGESTION_OPS_TOTAL.labels(operation=action, collection=col, status="completed").inc()
                    INGESTION_CHUNKS_CREATED.inc(chunks_count)
                    # Clean up temp file
                    if filepath and os.path.exists(filepath):
                        try:
                            os.unlink(filepath)
                        except Exception:
                            pass
                else:
                    queue_db.update_queue_status(op_id, "failed", completed_at=datetime.now().isoformat(), error=op_error)
                    failed_count += 1
                    INGRED_FAIL = INGESTION_OPS_FAILED_TOTAL.inc()
                    INGESTION_OPS_TOTAL.labels(operation=action, collection=col, status="failed").inc()
                
                summary_ops.append({
                    "id": op_id,
                    "document_id": doc_id,
                    "operation": action,
                    "status": "completed" if op_success else "failed",
                    "error": op_error
                })
                
        finally:
            # 2. Maintenance mode exit: Start vLLM
            if vllm_stopped:
                logger.info("Restarting vLLM container...")
                DockerManager.start_container(settings.VLLM_CONTAINER_NAME)
                # Wait for models route to be healthy
                await wait_for_vllm_ready()
                
            if gpu_maintenance:
                chat_gate.allow_chats()

        duration = time.time() - start_time
        completed_timestamp = datetime.now().isoformat()
        
        job_summary = {
            "total_processed": len(pending_ops),
            "success": success_count,
            "failed": failed_count,
            "chunks_created": chunks_total,
            "duration_seconds": round(duration, 2),
            "operations": summary_ops
        }

        job_status = "completed" if failed_count == 0 else "failed"
        
        # Save to database history
        queue_db.add_history(
            started_at=start_timestamp,
            completed_at=completed_timestamp,
            status=job_status,
            summary=job_summary,
            error=f"{failed_count} operations failed" if failed_count > 0 else None
        )

        # Update metrics
        INGESTION_DURATION.set(duration)
        INGESTION_QUEUE_SIZE.set(len(queue_db.get_queue(status="pending")))
        if job_status == "completed":
            INGESTION_LAST_SUCCESS.set(time.time())
        else:
            INGESTION_LAST_FAILURE.set(time.time())

        ingestion_status.is_processing = False
        ingestion_status.last_run_summary = job_summary
        
        logger.info(f"Ingestion job run complete: {job_summary}")
        return job_summary


# Scheduled worker loop task running at INGESTION_SCHEDULE
async def scheduled_sync_loop():
    logger.info(f"Ingestion worker background scheduler started. Schedule: {settings.INGESTION_SCHEDULE}")
    while True:
        try:
            h, m = map(int, settings.INGESTION_SCHEDULE.split(":"))
        except Exception:
            logger.error(f"Invalid INGESTION_SCHEDULE value: '{settings.INGESTION_SCHEDULE}'. Defaulting to 00:00.")
            h, m = 0, 0
            
        now = datetime.now()
        target = datetime.combine(now.date(), dt_time(h, m))
        if target <= now:
            # Schedule for tomorrow
            target = target + timedelta(days=1)
            
        seconds_to_wait = (target - now).total_seconds()
        logger.info(f"Next scheduled queue ingestion run in {seconds_to_wait:.1f} seconds (at {target})")
        
        try:
            await asyncio.sleep(seconds_to_wait)
            logger.info("Executing scheduled midnight queue ingestion...")
            await run_queue_ingestion()
        except asyncio.CancelledError:
            logger.info("Scheduled synchronization loop task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error occurred during scheduled synchronization loop: {e}")
            await asyncio.sleep(60) # Wait before retry
