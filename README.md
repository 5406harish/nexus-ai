# 🧠 Nexus AI — Hybrid RAG Knowledge Base

A modern **Hybrid RAG** system with semantic search + BM25, agentic reasoning using Claude, real-time streaming responses, and a clean web UI. Powered by Endee Vector Database.

## ✨ Features
- Hybrid Retrieval (Dense Embeddings + BM25)
- Agentic RAG with Anthropic Claude (tool calling)
- Real-time streaming responses
- Document ingestion (PDF, Markdown, Text)
- Beautiful Chat + Admin Interface
## ✨ What This Project Does

**Nexus** allows you to:

- Ingest documents (PDF, Markdown, Text) into an Endee vector index
- Perform **hybrid search** (dense semantic embeddings + BM25 keyword search)
- Ask natural language questions — Claude AI answers with accurate citations
- Use an **agentic workflow** where the AI intelligently decides when to retrieve information
- Get **real-time streaming** responses (token by token)
- Filter results by category, source, date, or relevance
- Manage everything through a clean web UI + Admin panel

> Hybrid RAG (Retrieval-Augmented Generation) system powered by [Endee](https://github.com/endee-io/endee) vector database, Ollama (local LLM), and BM25 hybrid search.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![Endee](https://img.shields.io/badge/Endee-Vector_DB-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![Docker](https://img.shields.io/badge/Docker-required-blue)

---

## 📌 What This Project Does

**Nexus** is a production-grade AI knowledge base that lets you:

- **Ingest** documents (text, PDF, markdown) into an Endee vector index
- **Search** using hybrid dense + BM25 (keyword) retrieval
- **Ask questions** in natural language — Ollama reads relevant chunks and answers with citations
- **Filter** by category, source, date, or relevance score
- **Streaming responses** — real-time token-by-token answers in the UI

project architecture---
nexus-rag/
├── backend/
│   ├── main.py              # FastAPI app — all API routes
│   ├── endee_client.py      # Endee SDK wrapper (hybrid index, upsert, query)
│   ├── embeddings.py        # Dense + BM25 sparse embedding pipeline
│   ├── rag_pipeline.py      # RAG orchestration with Ollama streaming
│   ├── ingestion.py         # Document chunking + ingestion
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Main app UI
│   ├── app.js               # Frontend logic
│   └── style.css            # Styles
├── scripts/
│   └── ingest_sample_data.py
├── docker-compose.yml
├── .env.example
└── README.md
Step-by-step Run Commands

Clone the repository
git clone https://github.com/5406harish/nexus-ai.git
cd nexus-ai
Setup Environment
cp .env.example .env→ Edit .env and add your ANTHROPIC_API_KEY.
Start Endee Vector Database
docker start endee-server(If the container is not created yet, use docker-compose up -d first)
Start the Backend
docker start nexus-backend
Open the Frontend
start nexus-rag/frontend/index.html
🔌 API Endpoints
Method	Endpoint	Description
POST	/api/ingest	Ingest documents into Endee
POST	/api/search	Hybrid semantic + BM25 search
POST	/api/chat	Streaming RAG chat (SSE)
GET	/api/index/stats	Index statistics
DELETE	/api/index/reset	Reset the index
GET	/api/health	Health check
🛠️ Configuration
Variable	Default	Description
OLLAMA_BASE_URL	http://host.docker.internal:11434	Ollama server URL
OLLAMA_MODEL	llama3.2	Model to use
ENDEE_BASE_URL	http://host.docker.internal:8080/api/v1	Endee server URL
ENDEE_INDEX_NAME	nexus_knowledge_base	Index name
EMBEDDING_MODEL	all-MiniLM-L6-v2	Sentence transformer model
CHUNK_SIZE	512	Token chunk size
CHUNK_OVERLAP	64
⚠️ Known Issues & Fixes Applied
endee==0.1.25 SDK checksum bug — SDK sends checksum=-1 which the server rejects. Fixed by using raw HTTP for index creation.
Python 3.14 incompatibility — No pre-built wheels for tiktoken, mmh3 on 3.14. Use the provided Docker setup (Python 3.10).
