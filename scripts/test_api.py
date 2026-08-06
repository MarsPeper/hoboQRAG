import argparse
import sys
import httpx

def main():
    parser = argparse.ArgumentParser(description="Manual API CLI Test Tool for hoboQRAG")
    parser.add_argument("--url", default="http://localhost:80", help="FastAPI Base API URL (default: http://localhost:80)")
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    print(f"Starting manual API test against: {base_url}")
    
    test_col = "api_manual_smoke_test_collection"
    
    try:
        # 1. List collections
        print("\n1. Listing collections...")
        resp = httpx.get(f"{base_url}/api/collections")
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        collections = resp.json().get("collections", [])
        print(f"Found collections: {collections}")
        
        # 2. Create a collection
        if test_col not in collections:
            print(f"\n2. Creating test collection '{test_col}'...")
            resp = httpx.post(f"{base_url}/api/collections", json={"name": test_col})
            assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
            print("Collection created successfully.")
        else:
            print(f"\n2. Test collection '{test_col}' already exists, skipping creation.")
            
        # 3. Upload a test document
        print("\n3. Uploading a test document...")
        doc_content = (
            "The default timeout for manual testing is 120 seconds.\n"
            "The test portal URL is dev-portal.example.com."
        )
        files = {"file": ("manual_test_doc.txt", doc_content.encode("utf-8"), "text/plain")}
        resp = httpx.post(f"{base_url}/api/collections/{test_col}/documents", files=files)
        assert resp.status_code == 201, f"HTTP {resp.status_code}: {resp.text}"
        print(f"Document uploaded. Chunks: {resp.json().get('chunk_count')}")
        
        # 4. Query the collection (using chat prompt)
        print("\n4. Running chat query against test collection...")
        chat_payload = {
            "prompt": "What is the default timeout for manual testing?",
            "collection_name": test_col,
            "top_k": 2
        }
        
        # Print responses in stream
        print("Response stream:")
        with httpx.stream("POST", f"{base_url}/api/chat", json=chat_payload) as r:
            assert r.status_code == 200, f"HTTP {r.status_code}"
            for chunk in r.iter_text():
                print(chunk, end="", flush=True)
        print("\n\nManual API check complete: SUCCESS")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nManual API check failed! Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
