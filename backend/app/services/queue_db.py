import sqlite3
import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

class QueueDatabase:
    def __init__(self, db_path: Path = settings.QUEUE_DB_PATH):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_queue (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        filepath TEXT,
                        status TEXT NOT NULL,
                        queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        queued_by TEXT NOT NULL,
                        scheduled_for TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        error TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_history (
                        id TEXT PRIMARY KEY,
                        started_at TIMESTAMP NOT NULL,
                        completed_at TIMESTAMP,
                        status TEXT NOT NULL,
                        summary TEXT,
                        error TEXT
                    )
                """)
            logger.info("SQLite ingestion queue tables initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
        finally:
            conn.close()

    def get_queue(self, status: str = None) -> List[Dict[str, Any]]:
        """Retrieves queue records, optionally filtered by status."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM ingestion_queue WHERE status = ? ORDER BY queued_at ASC",
                    (status,)
                )
            else:
                cursor.execute("SELECT * FROM ingestion_queue ORDER BY queued_at ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_queue_item(self, operation_id: str) -> Dict[str, Any] | None:
        """Retrieves a single queue item by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ingestion_queue WHERE id = ?", (operation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_pending_by_document(self, document_id: str, collection: str) -> Dict[str, Any] | None:
        """Gets a pending queue item for a specific document in a collection."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM ingestion_queue WHERE document_id = ? AND collection = ? AND status = 'pending'",
                (document_id, collection)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def insert_queue_item(self, collection: str, document_id: str, operation: str, filename: str, filepath: str, queued_by: str) -> str:
        """Inserts a new queue item."""
        conn = self._get_connection()
        op_id = str(uuid.uuid4())
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO ingestion_queue (id, collection, document_id, operation, filename, filepath, status, queued_by)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (op_id, collection, document_id, operation.upper(), filename, filepath, queued_by)
                )
            return op_id
        finally:
            conn.close()

    def update_queue_file(self, operation_id: str, filename: str, filepath: str):
        """Updates the physical file details of an existing enqueued item (e.g. on REPLACE update)."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE ingestion_queue SET filename = ?, filepath = ?, queued_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (filename, filepath, operation_id)
                )
        finally:
            conn.close()

    def delete_queue_item(self, operation_id: str):
        """Deletes a queue item completely."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM ingestion_queue WHERE id = ?", (operation_id,))
        finally:
            conn.close()

    def update_queue_status(self, operation_id: str, status: str, started_at: str = None, completed_at: str = None, error: str = None):
        """Updates status and timestamps of an enqueued operation."""
        conn = self._get_connection()
        try:
            with conn:
                query = "UPDATE ingestion_queue SET status = ?"
                params = [status]
                if started_at:
                    query += ", started_at = ?"
                    params.append(started_at)
                if completed_at:
                    query += ", completed_at = ?"
                    params.append(completed_at)
                if error:
                    query += ", error = ?"
                    params.append(error)
                query += " WHERE id = ?"
                params.append(operation_id)
                conn.execute(query, tuple(params))
        finally:
            conn.close()

    def add_history(self, started_at: str, completed_at: str, status: str, summary: Dict[str, Any], error: str = None) -> str:
        """Saves completed job execution run details to history."""
        conn = self._get_connection()
        job_id = str(uuid.uuid4())
        summary_str = json.dumps(summary)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO ingestion_history (id, started_at, completed_at, status, summary, error)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, started_at, completed_at, status, summary_str, error)
                )
            return job_id
        finally:
            conn.close()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves history of past ingestion jobs."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM ingestion_history ORDER BY started_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("summary"):
                    try:
                        d["summary"] = json.loads(d["summary"])
                    except Exception:
                        pass
                results.append(d)
            return results
        finally:
            conn.close()

    def resolve_conflicts_and_enqueue(self, collection: str, document_id: str, operation: str, filename: str, filepath: str, queued_by: str) -> str:
        """
        Enqueues an operation and resolves conflicts based on RAG queue deployment rules.
        """
        existing = self.get_pending_by_document(document_id, collection)
        operation = operation.upper()

        if not existing:
            # Enqueue directly
            return self.insert_queue_item(collection, document_id, operation, filename, filepath, queued_by)

        existing_op = existing["operation"]
        existing_id = existing["id"]

        if operation == "ADD":
            if existing_op == "ADD":
                raise ValueError(f"Conflict: Document '{document_id}' already has a pending ADD operation.")
            elif existing_op == "DELETE":
                # Convert DELETE to REPLACE
                self.delete_queue_item(existing_id)
                return self.insert_queue_item(collection, document_id, "REPLACE", filename, filepath, queued_by)
            elif existing_op == "REPLACE":
                raise ValueError(f"Conflict: Document '{document_id}' already has a pending REPLACE operation.")

        elif operation == "DELETE":
            if existing_op == "ADD":
                # Net zero change: cancel the pending ADD (and clean up temp file)
                self.delete_queue_item(existing_id)
                if existing["filepath"] and os.path.exists(existing["filepath"]):
                    try:
                        os.unlink(existing["filepath"])
                    except Exception:
                        pass
                return "CANCELLED_PENDING_ADD"
            elif existing_op == "REPLACE":
                # Cancel the REPLACE, convert to a pure DELETE
                self.delete_queue_item(existing_id)
                if existing["filepath"] and os.path.exists(existing["filepath"]):
                    try:
                        os.unlink(existing["filepath"])
                    except Exception:
                        pass
                return self.insert_queue_item(collection, document_id, "DELETE", "", "", queued_by)
            elif existing_op == "DELETE":
                # Keep existing DELETE
                return existing_id

        elif operation == "REPLACE":
            if existing_op == "ADD":
                # Update the pending ADD file content (replaces the add file payload)
                if existing["filepath"] and os.path.exists(existing["filepath"]):
                    try:
                        os.unlink(existing["filepath"])
                    except Exception:
                        pass
                self.update_queue_file(existing_id, filename, filepath)
                return existing_id
            elif existing_op == "REPLACE":
                # Overwrite the existing pending replacement file payload
                if existing["filepath"] and os.path.exists(existing["filepath"]):
                    try:
                        os.unlink(existing["filepath"])
                    except Exception:
                        pass
                self.update_queue_file(existing_id, filename, filepath)
                return existing_id
            elif existing_op == "DELETE":
                raise ValueError(f"Conflict: Document '{document_id}' is flagged for deletion and cannot be replaced.")

        raise ValueError(f"Unknown operation: {operation}")

# Singleton Instance
queue_db = QueueDatabase()
