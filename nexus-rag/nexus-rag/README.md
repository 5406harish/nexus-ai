# 🧠 Nexus — Advanced AI Knowledge Base with Endee Vector Database

> Hybrid RAG (Retrieval-Augmented Generation) system powered by [Endee](https://github.com/endee-io/endee) vector database, Claude AI, and BM25 hybrid search.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![Endee](https://img.shields.io/badge/Endee-Vector_DB-purple)
![Claude](https://img.shields.io/badge/Claude-Anthropic-orange)
![Docker](https://img.shields.io/badge/Docker-required-blue)

---

## 📌 What This Project Does

**Nexus** is a production-grade AI knowledge base that lets you:

- **Ingest** documents (text, PDF, markdown) into an Endee vector index
- **Search** using hybrid dense + BM25 (keyword) retrieval for best-of-both accuracy
- **Ask questions** in natural language — Claude AI reads relevant chunks and answers with citations
- **Filter** by category, source, date, or relevance score
- **Agentic workflow** — the AI decides *when* to search vs answer from context
- **Streaming responses** — real-time token-by-token answers in the UI
- **Admin panel** — manage indexes, view stats, re-ingest documents

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS)                  │
│         Chat UI · Search Panel · Admin Dashboard        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / SSE (streaming)
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend (Python)                │
│                                                         │
│  ┌─────────────┐  ┌────────────────┐  ┌─────────────┐  │
│  │  Ingestion  │  │  RAG Pipeline  │  │  Search API │  │
│  │  Pipeline   │  │  (Agentic)     │  │  Endpoint   │  │
│  └──────┬──────┘  └───────┬────────┘  └──────┬──────┘  │
│         │                 │                   │         │
│  ┌──────▼─────────────────▼───────────────────▼──────┐  │
│  │           Endee Client Layer                      │  │
│  │   Dense Embeddings · Sparse BM25 · Hybrid Query   │  │
│  └──────────────────────┬─────────────────────────────┘  │
└─────────────────────────┼──────────────────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │        Endee Vector DB          │
         │  (Docker · Port 8080)           │
         │  HNSW Index + BM25 Sparse       │
         └─────────────────────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │      Anthropic Claude API       │
         │  (Streaming · Tool Use)         │
         └─────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/nexus-rag.git
cd nexus-rag
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Start Endee vector database

```bash
docker run -d \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest

# Verify it's running
curl http://localhost:8080/api/v1/index/list
```

### 4. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Ingest sample knowledge base

```bash
python ../scripts/ingest_sample_data.py
```

### 6. Start the backend

```bash
uvicorn main:app --reload --port 7860
```

### 7. Open the UI

Open `frontend/index.html` in your browser (or use Live Server in VS Code).

---

## 📁 Project Structure

```
nexus-rag/
├── backend/
│   ├── main.py              # FastAPI app — all API routes
│   ├── endee_client.py      # Endee SDK wrapper (hybrid index, upsert, query)
│   ├── embeddings.py        # Dense + BM25 sparse embedding pipeline
│   ├── rag_pipeline.py      # Agentic RAG orchestration with Claude
│   ├── ingestion.py         # Document chunking + ingestion
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Main app UI
│   ├── app.js               # Frontend logic
│   └── style.css            # Styles
├── scripts/
│   └── ingest_sample_data.py  # Loads 50+ sample tech documents
├── data/                    # (auto-created) local document storage
├── docker-compose.yml       # Run Endee via compose
├── .env.example
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest` | Ingest documents into Endee |
| `POST` | `/api/search` | Hybrid semantic + BM25 search |
| `POST` | `/api/chat` | Streaming RAG chat (SSE) |
| `GET`  | `/api/index/stats` | Index statistics |
| `DELETE` | `/api/index/reset` | Reset the index |
| `GET`  | `/api/health` | Health check |

### Chat Request Example

```json
POST /api/chat
{
  "message": "What are the best practices for designing microservices?",
  "conversation_history": [],
  "filters": { "category": "architecture" },
  "top_k": 5,
  "hybrid_alpha": 0.7
}
```

### Search Request Example

```json
POST /api/search
{
  "query": "vector database performance benchmarks",
  "top_k": 10,
  "filters": { "category": "databases" },
  "hybrid_alpha": 0.6
}
```

---

## 🧬 Key Features Explained

### Hybrid Search (Dense + BM25)
Endee's hybrid search combines:
- **Dense vectors** (384-dim from `all-MiniLM-L6-v2`) — captures semantic meaning and synonyms
- **BM25 sparse vectors** — captures exact keyword matches

The `hybrid_alpha` parameter (0.0–1.0) controls the blend. `0.7` = 70% semantic + 30% keyword.

### Agentic RAG Pipeline
The AI pipeline uses Claude's tool-use API:
1. **Decide** — Does the question need a knowledge base lookup?
2. **Search** — Execute hybrid search with optimal parameters
3. **Rerank** — Score chunks by relevance + recency
4. **Generate** — Stream the answer with inline citations

### Chunking Strategy
Documents are split with:
- Chunk size: 512 tokens (configurable)
- Overlap: 64 tokens
- Sentence-boundary aware (no mid-sentence cuts)

---

## 🛠️ Configuration (`.env`)

```env
ANTHROPIC_API_KEY=sk-ant-...
ENDEE_BASE_URL=http://localhost:8080/api/v1
ENDEE_AUTH_TOKEN=          # Leave empty if not set
ENDEE_INDEX_NAME=nexus_knowledge_base
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

---

## 📊 Performance

| Operation | Latency |
|-----------|---------|
| Document ingestion (1000 chunks) | ~15s |
| Hybrid search (50k docs) | <20ms |
| RAG end-to-end (first token) | ~800ms |
| Full streaming response | 2–5s |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR

---

## 📄 License

MIT — built with ❤️ on top of [Endee](https://github.com/endee-io/endee) (Apache-2.0)
