import requests

BASE = "http://host.docker.internal:8080/api/v1"

# Try creating index with different checksums
for checksum in [0, 1, 42, 100, 999, -1, 12345]:
    data = {
        "index_name": "nexus_knowledge_base",
        "dim": 384,
        "space_type": "cosine",
        "M": 16,
        "ef_con": 128,
        "checksum": checksum,
        "precision": "int8",
        "sparse_model": "endee_bm25"
    }
    r = requests.post(f"{BASE}/index/create", json=data)
    print(f"checksum={checksum}: {r.status_code} {r.text[:100]}")
    if r.status_code == 200:
        break
