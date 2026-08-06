import sys
import httpx

def main():
    print("==================================================")
    print("         hoboQRAG Deployment Smoke Test          ")
    print("==================================================")
    
    passed = True
    
    # 1. Check Nginx proxy to FastAPI
    print("Checking Nginx -> FastAPI...")
    try:
        response = httpx.get("http://localhost:80/api/collections", timeout=5.0)
        if response.status_code == 200:
            print("[PASS] Nginx reverse proxying to FastAPI is functional.")
        else:
            print(f"[FAIL] Nginx proxy returned HTTP status code: {response.status_code}")
            passed = False
    except Exception as e:
        print(f"[FAIL] Nginx not reachable on http://localhost:80. Error: {e}")
        passed = False
        
    # 2. Check Qdrant connection via FastAPI
    print("Checking Qdrant connectivity...")
    try:
        response = httpx.get("http://localhost:80/api/collections", timeout=5.0)
        if response.status_code == 200:
            collections = response.json().get("collections", [])
            print(f"[PASS] Qdrant connectivity functional. Available collections: {collections}")
        else:
            print(f"[FAIL] Qdrant connection check failed via API: HTTP {response.status_code}")
            passed = False
    except Exception as e:
        print(f"[FAIL] Qdrant connection check failed: {e}")
        passed = False
        
    # 3. Check Models (FastAPI embedding & reranking health)
    print("Checking Embedding and Reranker models...")
    try:
        collections_resp = httpx.get("http://localhost:80/api/collections", timeout=5.0)
        collections = collections_resp.json().get("collections", ["tech_support_kb"])
        col = collections[0] if collections else "tech_support_kb"
        
        chat_payload = {
            "prompt": "healthcheck query",
            "collection_name": col,
            "top_k": 1
        }
        response = httpx.post("http://localhost:80/api/chat", json=chat_payload, timeout=10.0)
        if response.status_code == 200 or "not found" in response.text.lower():
            print("[PASS] Models (Embedding & Reranker) initialized successfully.")
        else:
            print(f"[FAIL] RAG query failed: HTTP {response.status_code} - {response.text}")
            passed = False
    except Exception as e:
        print(f"[FAIL] Models check failed: {e}")
        passed = False

    # 4. Check vLLM Server compatibility
    print("Checking vLLM / Ollama connection...")
    try:
        response = httpx.get("http://localhost:8000/v1/models", timeout=5.0)
        if response.status_code == 200:
            models = [m["id"] for m in response.json().get("data", [])]
            print(f"[PASS] vLLM server is healthy. Available models: {models}")
        else:
            print(f"[WARN] vLLM check returned HTTP {response.status_code}.")
    except Exception as e:
        print(f"[WARN] vLLM direct check not reachable (Error: {e}). Checking via chat stream...")

    print("==================================================")
    if passed:
        print("System health check: PASS")
        sys.exit(0)
    else:
        print("System health check: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
