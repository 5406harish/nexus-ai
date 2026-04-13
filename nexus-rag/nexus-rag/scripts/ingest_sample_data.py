#!/usr/bin/env python3
"""
scripts/ingest_sample_data.py

Populates the Nexus knowledge base with a rich set of technical documents
covering: vector databases, AI/ML, software architecture, distributed systems,
devops, and more.

Run from the project root:
  python scripts/ingest_sample_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import logging
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from endee_client import EndeeClient
from embeddings   import EmbeddingPipeline
from ingestion    import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Knowledge base documents
# ──────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    # ── Vector Databases ──────────────────────────────────────────────
    {
        "title": "Introduction to Vector Databases",
        "category": "databases",
        "source": "Nexus Knowledge Base",
        "author": "Nexus Team",
        "text": """
Vector databases are purpose-built data stores optimized for storing and querying
high-dimensional vectors. Unlike traditional databases that excel at exact matches,
vector databases perform approximate nearest-neighbor (ANN) searches that find the
most semantically similar items to a query.

A vector is a list of numbers (floats) representing an object — a sentence,
an image, a product, or any entity that can be encoded. Machine learning models
called embedding models convert raw data into these vectors. Semantically similar
items cluster close together in the vector space.

Key capabilities of vector databases:
• ANN indexing: HNSW, IVF, PQ, and other graph/tree-based indexes
• Metadata filtering: combine vector similarity with boolean/range filters
• Hybrid search: blend dense (semantic) and sparse (keyword/BM25) signals
• Scalability: handle billions of vectors on a single or distributed node
• Persistence: durable storage with WAL (write-ahead logging)

Popular vector databases include Endee, Pinecone, Weaviate, Qdrant, Milvus,
PGVector (PostgreSQL extension), and FAISS (library, not a full DB).
""",
    },
    {
        "title": "HNSW: Hierarchical Navigable Small World Graphs",
        "category": "databases",
        "source": "Research Summary",
        "author": "Nexus Team",
        "text": """
Hierarchical Navigable Small World (HNSW) is the dominant ANN indexing algorithm
used in modern vector databases. It builds a multi-layer graph where each layer is
a random subset of the layer below, and the bottom layer contains all points.

Construction (insert):
1. Start at the entry point of the top layer
2. Greedily navigate toward the new point using nearest-neighbor links
3. Insert the point into each layer from top down to a randomly chosen bottom layer
4. Connect it to its M nearest neighbors in each layer

Search (query):
1. Enter at the top layer's entry point
2. Greedily navigate down each layer
3. At the bottom layer, collect the ef_search nearest candidates
4. Return the top-K results

Key parameters:
• M: number of bidirectional links per node (controls graph connectivity; higher = better recall, more memory)
• ef_construction: beam width during build (higher = better quality, slower build)
• ef_search: beam width during query (higher = better recall, slower query)

HNSW achieves O(log N) search complexity with high recall (>0.99) at millisecond latency,
making it far superior to brute-force O(N) search for large datasets.
""",
    },
    {
        "title": "Endee Vector Database: Architecture and Features",
        "category": "databases",
        "source": "Endee Documentation",
        "author": "Endee Labs",
        "text": """
Endee (nD) is a high-performance open-source vector database designed to handle
up to 1 billion vectors on a single node. Its key differentiators are:

Architecture:
• Written in C++20 with SIMD optimizations (AVX2, AVX512, NEON, SVE2)
• HNSW-based indexing with configurable M and ef_construction parameters
• INT8 and FP16 quantization for 4–8× memory reduction with minimal recall loss
• Apache License 2.0; also available as a managed cloud service (endee.io)

Hybrid Search:
Endee supports combining dense (semantic) and sparse (BM25) vectors in a single
index. The sparse model is set at index creation time with sparse_model="endee_bm25".
Documents store TF (term frequency) weights; the server applies IDF (inverse document
frequency) on-the-fly using pre-computed corpus statistics.

Python SDK:
  pip install endee endee-model
  from endee import Endee, Precision
  client = Endee()
  client.create_index(name="my_index", dimension=384, space_type="cosine",
                      sparse_model="endee_bm25")

Filter support:
  $eq, $in, $range operators on metadata fields.
  Filters are applied post-ANN to preserve recall.

REST API: Endee exposes all operations over HTTP/JSON at port 8080.
Docker: docker run -p 8080:8080 -v ./data:/data endeeio/endee-server:latest
""",
    },
    {
        "title": "BM25: Best Match 25 for Keyword Retrieval",
        "category": "databases",
        "source": "IR Research",
        "author": "Nexus Team",
        "text": """
BM25 (Best Match 25) is a probabilistic information retrieval model that scores
documents based on the query terms they contain. It is the backbone of most search
engines (Elasticsearch, Solr, Lucene) and is used as the sparse component in
hybrid vector search.

Formula:
  Score(D, Q) = Σ IDF(qi) × [ TF(qi, D) × (k1 + 1) ] / [ TF(qi, D) + k1 × (1 - b + b × |D|/avgdl) ]

Where:
  TF(qi, D)  = term frequency of query term qi in document D
  IDF(qi)    = log[ (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 ]
  |D|        = document length (word count)
  avgdl      = average document length across corpus
  k1 ≈ 1.2   = term saturation parameter
  b  ≈ 0.75  = length normalization parameter

Why BM25 complements dense vectors:
  Dense vectors capture semantic similarity but can miss exact technical terms,
  product codes, proper nouns, and rare vocabulary. BM25 excels precisely where
  dense models struggle. Hybrid search (α × dense + (1-α) × BM25) consistently
  outperforms either approach alone on retrieval benchmarks.
""",
    },
    {
        "title": "Vector Quantization: INT8 and Binary Compression",
        "category": "databases",
        "source": "ML Systems Research",
        "author": "Nexus Team",
        "text": """
Vector quantization reduces the memory footprint of stored embeddings at the cost
of a small accuracy loss. Modern vector databases offer several quantization modes:

FP32 (no quantization): 4 bytes per dimension — baseline accuracy, maximum memory.
FP16: 2 bytes per dimension — ~2× compression, negligible recall loss on cosine space.
INT8: 1 byte per dimension — ~4× compression, ~0.5–1% recall loss, significant speedup
      via SIMD integer arithmetic (8-bit dot products are much faster than float ops).
Binary: 1 bit per dimension — ~32× compression, 5–10% recall loss, suitable for very
        large-scale approximate retrieval as a pre-filter.

Endee uses INT8 by default (Precision.INT8) which provides the best accuracy/memory
tradeoff for most use cases. The stored quantized vectors are dequantized on retrieval
so the similarity scores remain accurate.

When to use quantization:
  • INT8: Always (default) — excellent recall, 4× memory savings
  • Binary: Pre-filtering stage for billion-scale indexes
  • FP32: Research/precision-critical applications where memory is not a constraint
""",
    },

    # ── RAG & LLMs ────────────────────────────────────────────────────
    {
        "title": "Retrieval-Augmented Generation (RAG): Complete Guide",
        "category": "ai-ml",
        "source": "AI Research Digest",
        "author": "Nexus Team",
        "text": """
Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large
language models (LLMs) by providing them with relevant, up-to-date context retrieved
from an external knowledge base at inference time.

Why RAG?
  LLMs have a training knowledge cutoff and can hallucinate facts. RAG grounds
  the model's response in retrieved evidence, improving factuality and enabling
  the model to answer questions about proprietary or recent information.

Standard RAG pipeline:
  1. Index: chunk documents → embed → store in vector DB
  2. Retrieve: embed query → ANN search → return top-K chunks
  3. Augment: inject chunks into the LLM prompt as context
  4. Generate: LLM produces a grounded answer with citations

Advanced RAG variants:
  • Hybrid RAG: dense + sparse (BM25) retrieval for better recall
  • HyDE (Hypothetical Document Embeddings): generate a hypothetical answer,
    embed it, use that embedding to search (improves query-document alignment)
  • RAG-Fusion: run multiple query rewrites, merge results with RRF
  • Self-RAG: the LLM decides when to retrieve and critiques its own output
  • Agentic RAG: multi-step retrieval with tool use and chain-of-thought

Evaluation metrics:
  • Faithfulness: is the answer supported by the retrieved context?
  • Answer relevance: does the answer address the question?
  • Context recall: did retrieval find all relevant chunks?
  • Context precision: are retrieved chunks relevant (no noise)?
""",
    },
    {
        "title": "Prompt Engineering for RAG Systems",
        "category": "ai-ml",
        "source": "Anthropic Research",
        "author": "Nexus Team",
        "text": """
Effective prompting is critical for RAG quality. The system prompt controls how the
LLM uses retrieved context, handles uncertainty, and cites sources.

Core RAG system prompt structure:
  1. Role definition: "You are an expert assistant with access to a knowledge base"
  2. Context injection: "Use ONLY the provided documents to answer"
  3. Citation instructions: "Cite the document title when using its content"
  4. Uncertainty handling: "If the answer is not in the documents, say so"
  5. Format guidance: "Use markdown, bullet points for lists"

Anti-patterns to avoid:
  • Over-constraining: telling the model to ONLY use context prevents it from
    applying reasoning to synthesize an answer
  • Under-constraining: no citation instructions leads to hallucination
  • Context stuffing: injecting too many chunks degrades answer quality
    (sweet spot: 3–6 high-quality chunks per query)

Chunk prompt format:
  [1] Source: {title} | Category: {category} | Score: {similarity:.3f}
  {chunk_text}

Claude-specific tips:
  • Claude respects "cite your sources" instructions reliably
  • Claude handles long contexts well (200K token window)
  • Use tool use / function calling for agentic retrieval loops
  • Streaming (SSE) dramatically improves perceived latency
""",
    },
    {
        "title": "Chunking Strategies for RAG Pipelines",
        "category": "ai-ml",
        "source": "RAG Best Practices",
        "author": "Nexus Team",
        "text": """
Document chunking is one of the most impactful design decisions in a RAG system.
The chunk is the unit of retrieval — it must be semantically coherent and contain
enough context to be useful to the LLM.

Fixed-size chunking:
  Split every N tokens with O overlap. Simple and predictable.
  Risk: cuts mid-sentence, losing semantic coherence.

Sentence-boundary chunking:
  Split on sentence endings (., !, ?). More coherent chunks.
  Implementation: spaCy, NLTK sentencizer, or regex.

Semantic chunking:
  Embed each sentence and split where the embedding similarity drops sharply.
  Produces chunks that are semantically self-contained.
  Cost: requires embedding every sentence during ingestion.

Recursive character text splitting:
  Prefer splitting on paragraphs → sentences → words → characters.
  LangChain's RecursiveCharacterTextSplitter uses this approach.

Parent-child chunking:
  Store small chunks for retrieval (512 tokens) but return the parent chunk
  (2048 tokens) to the LLM for more context. Best of both worlds.

Recommended settings:
  • General knowledge base: 512 tokens, 64 token overlap
  • Legal/technical documents: 256 tokens, 32 token overlap
  • Books/long-form: 1024 tokens, 128 token overlap
""",
    },
    {
        "title": "Agentic AI: Tool Use and Multi-Step Reasoning",
        "category": "ai-ml",
        "source": "AI Architecture Guide",
        "author": "Nexus Team",
        "text": """
Agentic AI systems give LLMs tools (functions) they can call autonomously to
gather information, take actions, and iteratively refine their output before
producing a final answer.

Tool use (function calling):
  The LLM is given a JSON schema describing available tools. When it decides to
  use a tool, it returns a structured tool_use block instead of text. The
  application executes the function and feeds the result back to the LLM.

Agentic RAG pattern:
  1. User asks a question
  2. Claude decides: does this need a KB search?
  3. If yes → emits search_knowledge_base tool call
  4. Application executes Endee hybrid search
  5. Results injected back as tool_result
  6. Claude may search again (different query) or proceed to answer
  7. Final answer generated with citations

Benefits over naive RAG:
  • The LLM can reformulate the query for better recall
  • Multi-hop retrieval: search → analyze → search again
  • Conditional retrieval: skip search for simple conversational replies
  • Self-verification: the LLM can check if retrieved context is sufficient

Anthropic's tool use API:
  Pass tools=[...] to messages.create()
  Set stop_reason == "tool_use" detection in streaming loop
  Inject tool_result into next messages[] turn
""",
    },
    {
        "title": "Embedding Models Comparison: all-MiniLM vs BGE vs OpenAI",
        "category": "ai-ml",
        "source": "Embedding Benchmarks",
        "author": "Nexus Team",
        "text": """
Choosing the right embedding model significantly impacts RAG retrieval quality.

all-MiniLM-L6-v2 (sentence-transformers):
  Dimensions: 384 | Size: 22MB | Latency: ~5ms/batch
  Pros: fast, small, good all-around performance
  Cons: lower accuracy than larger models on specialized tasks
  Best for: real-time applications, edge deployment, general knowledge bases

BGE-M3 (BAAI):
  Dimensions: 1024 | Size: 570MB | Latency: ~30ms/batch
  Pros: state-of-the-art multilingual, supports dense+sparse+colbert in one model
  Cons: larger, slower, requires more memory
  Best for: multilingual knowledge bases, high-accuracy requirements

text-embedding-ada-002 (OpenAI):
  Dimensions: 1536 | Size: API-only | Latency: ~100ms (API)
  Pros: excellent quality, maintained by OpenAI
  Cons: requires API call (latency + cost), data leaves your infrastructure

text-embedding-3-large (OpenAI):
  Dimensions: 3072 (configurable) | Matryoshka embedding support
  Best for: highest accuracy, budget not a concern

MTEB benchmark top performers (as of 2024):
  1. voyage-3-large (Voyage AI)
  2. text-embedding-3-large (OpenAI)
  3. GTE-Qwen2-7B (Alibaba)
  4. BGE-M3 (BAAI)
  5. all-MiniLM-L6-v2 (reasonable baseline, fastest)
""",
    },

    # ── Software Architecture ──────────────────────────────────────────
    {
        "title": "Microservices Architecture: Principles and Patterns",
        "category": "architecture",
        "source": "Software Architecture Guide",
        "author": "Nexus Team",
        "text": """
Microservices architecture decomposes a monolithic application into a collection of
small, independently deployable services, each owning its data and exposing APIs.

Core principles:
  • Single Responsibility: each service does one thing well
  • Decentralized data: each service owns its database (no shared DB)
  • API-first: services communicate via REST, gRPC, or async messaging
  • Failure isolation: one service failing doesn't cascade to others
  • Independent deployment: deploy any service without coordinating with others

Communication patterns:
  Synchronous: REST (JSON/HTTP), gRPC (Protobuf/HTTP2)
  Asynchronous: Kafka, RabbitMQ, SQS (event-driven, decoupled)

Data patterns:
  • Saga pattern: distributed transactions via event chains
  • CQRS: separate read/write models
  • Event sourcing: store events as the source of truth
  • API Gateway: single entry point for all client traffic

When to use microservices:
  ✓ Large teams (Conway's Law: system mirrors org structure)
  ✓ Different scaling requirements per component
  ✓ Independent deployment velocity needed
  ✗ Small teams / startups (overhead outweighs benefits)
  ✗ Simple CRUD applications (monolith is simpler)
""",
    },
    {
        "title": "Event-Driven Architecture with Apache Kafka",
        "category": "architecture",
        "source": "Distributed Systems Guide",
        "author": "Nexus Team",
        "text": """
Apache Kafka is a distributed event streaming platform designed for high-throughput,
fault-tolerant, real-time data pipelines and event-driven architectures.

Core concepts:
  Topic: a named stream of records (like a database table for events)
  Partition: a topic is split into ordered, immutable partitions
  Offset: position of a message within a partition
  Producer: writes events to topics
  Consumer: reads events from topics (tracks own offset)
  Consumer Group: multiple consumers sharing work (each partition → one consumer)
  Broker: a Kafka server (typically 3–5 in a cluster)

Kafka guarantees:
  • At-least-once delivery by default
  • Exactly-once semantics with transactions + idempotent producers
  • Message ordering within a partition
  • Retention: configurable (time-based or size-based, default 7 days)

Common patterns:
  Event sourcing: emit events for every state change
  CQRS + Kafka: events drive read-model updates asynchronously
  Stream processing: Kafka Streams / Flink / Spark Structured Streaming
  Change Data Capture (CDC): stream DB changes via Debezium → Kafka

Performance: Kafka handles millions of messages per second per broker.
Latency: ~5ms end-to-end for typical producer → consumer path.
""",
    },
    {
        "title": "API Design Best Practices: REST and gRPC",
        "category": "architecture",
        "source": "API Design Guide",
        "author": "Nexus Team",
        "text": """
Well-designed APIs are stable, intuitive, and evolvable. Here are the key
principles for both REST and gRPC APIs.

REST API best practices:
  • Use nouns for resources: /api/users, /api/documents (not /getUser)
  • HTTP verbs: GET (read), POST (create), PUT/PATCH (update), DELETE (remove)
  • Status codes: 200 OK, 201 Created, 400 Bad Request, 404 Not Found, 500 Server Error
  • Versioning: /api/v1/... in the URL or Accept: application/vnd.api+json;version=1 header
  • Pagination: cursor-based (after=<token>) or offset-based (?page=2&limit=20)
  • HATEOAS: responses include links to related resources (optional but recommended)
  • Rate limiting: 429 Too Many Requests with Retry-After header

gRPC advantages over REST:
  • Protobuf binary encoding: 3–10× smaller payload than JSON
  • Streaming: server-side, client-side, and bidirectional streaming
  • Strongly typed contracts via .proto files
  • Auto-generated client/server code in 10+ languages
  • HTTP/2: multiplexing, header compression, push

gRPC use cases: inter-service communication, real-time streaming, mobile-to-server.
REST use cases: public APIs, browser clients, simple CRUD, webhooks.
""",
    },

    # ── DevOps & Infrastructure ────────────────────────────────────────
    {
        "title": "Docker and Container Orchestration",
        "category": "devops",
        "source": "DevOps Handbook",
        "author": "Nexus Team",
        "text": """
Docker packages applications and their dependencies into portable containers that
run consistently across environments. Kubernetes orchestrates containers at scale.

Docker core concepts:
  Image: read-only template (layers from Dockerfile)
  Container: running instance of an image
  Registry: image storage (Docker Hub, ECR, GCR, GHCR)
  Dockerfile: instructions to build an image
  docker-compose: multi-container local orchestration

Dockerfile best practices:
  • Use official base images (python:3.11-slim, not python:latest)
  • Multi-stage builds to minimize final image size
  • Copy requirements.txt first (cache dependencies layer)
  • Run as non-root user (USER appuser)
  • Use .dockerignore to exclude node_modules, .git, etc.

Kubernetes concepts:
  Pod: smallest deployable unit (1+ containers sharing network)
  Deployment: manages Pod replicas + rolling updates
  Service: stable DNS name and load balancer for Pods
  ConfigMap / Secret: inject config without rebuilding images
  Ingress: HTTP routing to Services from outside the cluster
  PersistentVolumeClaim: durable storage for stateful workloads

Endee in Docker:
  docker run -p 8080:8080 -v ./data:/data endeeio/endee-server:latest
""",
    },
    {
        "title": "CI/CD Pipeline Design with GitHub Actions",
        "category": "devops",
        "source": "DevOps Handbook",
        "author": "Nexus Team",
        "text": """
CI/CD (Continuous Integration / Continuous Deployment) automates the path from
code commit to production deployment. GitHub Actions is the most popular CI/CD
platform for open-source projects.

CI pipeline stages:
  1. Trigger: push to main / pull_request
  2. Checkout: git checkout
  3. Install: pip install -r requirements.txt
  4. Lint: ruff check . / eslint
  5. Test: pytest / jest
  6. Build: docker build
  7. Push: docker push to registry

CD pipeline stages:
  1. Pull new image on server (SSH or rolling deployment)
  2. Health check: curl /api/health until 200
  3. Smoke test: basic API calls
  4. Traffic shift: blue/green or canary deployment

GitHub Actions example (.github/workflows/ci.yml):
  on: [push]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.11" }
        - run: pip install -r requirements.txt
        - run: pytest

Best practices:
  • Cache dependencies between runs (actions/cache)
  • Use secrets for API keys (never hardcode)
  • Matrix testing across Python/Node versions
  • Require PR approval before merging to main
""",
    },

    # ── Machine Learning ───────────────────────────────────────────────
    {
        "title": "Transformer Architecture: Attention and Self-Attention",
        "category": "ai-ml",
        "source": "Deep Learning Guide",
        "author": "Nexus Team",
        "text": """
The Transformer architecture (Vaswani et al., 2017 — "Attention Is All You Need")
is the foundation of all modern LLMs. It replaced recurrent networks with a
fully attention-based approach enabling massive parallelism.

Self-attention mechanism:
  Given input sequence X, compute:
    Q = X × W_Q   (queries)
    K = X × W_K   (keys)
    V = X × W_V   (values)
    Attention(Q,K,V) = softmax(QK^T / √d_k) × V

  Each token attends to all other tokens, capturing long-range dependencies
  without the vanishing gradient problem of RNNs.

Multi-head attention:
  Run h parallel attention heads with different weight matrices.
  Concatenate outputs and project: MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
  Each head learns different relationship patterns.

Architecture components:
  Encoder: processes input (BERT-style models for embeddings)
  Decoder: generates output auto-regressively (GPT-style)
  Encoder-Decoder: seq2seq tasks (T5, mT5)

Positional encoding:
  Transformers have no inherent sense of position.
  Solution: add sinusoidal or learned positional embeddings to token embeddings.
  Modern LLMs use RoPE (Rotary Positional Encoding) for better length generalization.

Scaling laws: model capability scales predictably with parameters, data, and compute.
""",
    },
    {
        "title": "Fine-Tuning vs RAG: When to Use Each",
        "category": "ai-ml",
        "source": "LLM Strategy Guide",
        "author": "Nexus Team",
        "text": """
Two main approaches to customize LLMs for domain-specific tasks: fine-tuning
and RAG. They serve different purposes and are often combined.

Fine-tuning:
  What: update model weights on domain-specific data
  When: changing behavior/style/format, not injecting facts
  Cost: GPU compute, labeled data, time (~hours to days)
  Best for: coding assistants, classification, structured output, tone/style

RAG:
  What: inject retrieved context at inference time, no weight updates
  When: grounding answers in specific documents, facts, or recent data
  Cost: embedding + vector DB infrastructure (cheap per query)
  Best for: question answering, documentation search, customer support

Fine-tuning fails at knowledge injection:
  Fine-tuned models hallucinate when asked about specific facts.
  Knowledge is distributed across billions of weights — hard to update.
  RAG is strictly superior for factual retrieval use cases.

Combine them (best practice):
  1. Fine-tune for format/behavior (output JSON, be concise, follow citations)
  2. RAG for factual grounding (retrieve relevant chunks per query)

Decision flowchart:
  New facts needed? → RAG
  Behavior change needed? → Fine-tuning
  Both? → Fine-tune + RAG
""",
    },
    {
        "title": "LLM Evaluation: Benchmarks and Metrics",
        "category": "ai-ml",
        "source": "AI Evaluation Guide",
        "author": "Nexus Team",
        "text": """
Evaluating LLMs requires multiple metrics across different dimensions. No single
benchmark captures all aspects of model quality.

Reasoning benchmarks:
  MMLU: 57-subject multiple-choice test covering STEM, humanities, social sciences
  GSM8K: grade school math word problems (tests multi-step reasoning)
  HumanEval: Python coding problems (pass@k metric)
  MATH: competition-level math problems

RAG-specific evaluation (RAGAS framework):
  Faithfulness: fraction of answer claims supported by retrieved context
    = |supported claims| / |total claims in answer|
  Answer Relevancy: cosine similarity between question and answer embedding
  Context Recall: fraction of ground-truth answer supported by context
  Context Precision: fraction of retrieved chunks that are relevant

Human evaluation:
  Likert scale (1–5) ratings for: helpfulness, accuracy, harmlessness, style
  Side-by-side comparisons (model A vs B — which is better?)
  MT-Bench: LLM-as-judge using GPT-4 to score multiturn conversations

Red-teaming:
  Adversarial testing for jailbreaks, harmful outputs, prompt injection
  PAIR (Prompt Automatic Iterative Refinement) for automated attacks

LLM-as-judge:
  Use a strong model (Claude 3 Opus, GPT-4o) to evaluate weaker models.
  Cheap, scalable, correlates well with human judgments (0.8+ Spearman).
""",
    },

    # ── Data Engineering ───────────────────────────────────────────────
    {
        "title": "Data Pipeline Architecture: Batch vs Stream Processing",
        "category": "data-engineering",
        "source": "Data Engineering Handbook",
        "author": "Nexus Team",
        "text": """
Data pipelines move and transform data from sources to destinations. The choice
between batch and stream processing depends on latency requirements.

Batch processing:
  Process large volumes of data at scheduled intervals (hourly, daily)
  Tools: Apache Spark, dbt, Pandas, BigQuery, Snowflake
  Latency: minutes to hours
  Best for: reporting, ML training, data warehouse ETL

Stream processing:
  Process data as it arrives, event by event or in micro-batches
  Tools: Apache Kafka + Flink, Spark Structured Streaming, Kinesis
  Latency: milliseconds to seconds
  Best for: real-time dashboards, fraud detection, alerting

Lambda architecture:
  Batch layer: accurate but slow (reprocesses all historical data)
  Speed layer: fast but approximate (processes recent data only)
  Serving layer: merges both for queries

Kappa architecture:
  Single streaming pipeline handles both real-time and historical data
  Replay the stream from beginning for historical reprocessing
  Simpler than Lambda but requires replayable storage (Kafka)

Modern data stack:
  Ingestion: Fivetran, Airbyte
  Warehouse: Snowflake, BigQuery, Databricks Lakehouse
  Transform: dbt (SQL-based transformations with version control)
  Orchestration: Airflow, Prefect, Dagster
  Reverse ETL: Census, Hightouch (warehouse → CRM/Ads)
""",
    },
    {
        "title": "Python Performance Optimization Techniques",
        "category": "programming",
        "source": "Python Best Practices",
        "author": "Nexus Team",
        "text": """
Python's Global Interpreter Lock (GIL) limits CPU parallelism, but many
optimization techniques can dramatically improve performance.

Profiling first:
  cProfile / py-spy: identify bottlenecks before optimizing
  memory_profiler: find memory leaks and peak usage

NumPy vectorization:
  Replace Python loops with NumPy array operations (100–1000× faster)
  Bad:  [x**2 for x in range(1000000)]
  Good: np.arange(1000000) ** 2

Caching:
  functools.lru_cache: memoize pure functions
  joblib.Memory: cache to disk for expensive computations
  Redis: distributed cache for multi-process applications

Concurrency:
  threading: good for I/O-bound tasks (API calls, DB queries)
  multiprocessing: CPU-bound tasks (bypasses GIL)
  asyncio: high-concurrency I/O (thousands of simultaneous connections)
  concurrent.futures: high-level interface for both

FastAPI + async:
  Use async def for route handlers that make I/O calls
  Use httpx.AsyncClient instead of requests for async HTTP
  Use asyncpg instead of psycopg2 for async PostgreSQL

NumPy + SIMD:
  NumPy uses BLAS/LAPACK with AVX2/AVX512 automatically
  Use float32 instead of float64 for 2× memory, same speed
  Batch operations: process many samples at once, not one-by-one
""",
    },

    # ── Security ──────────────────────────────────────────────────────
    {
        "title": "API Security: Authentication and Authorization",
        "category": "security",
        "source": "Security Best Practices",
        "author": "Nexus Team",
        "text": """
Securing APIs requires multiple layers of defense. Authentication verifies identity;
authorization determines what an authenticated entity can do.

Authentication methods:
  API Keys: simple, static tokens. Easy to implement, hard to rotate.
    Store as hash in DB, never log in plaintext.
    Pass in Authorization header: Authorization: Bearer sk-...
  JWT (JSON Web Tokens): self-contained, signed tokens.
    Header.Payload.Signature (base64 encoded, signed with HMAC or RSA)
    Stateless (no DB lookup) but cannot be revoked before expiry.
    Keep expiry short (15min) + use refresh tokens.
  OAuth 2.0: delegated authorization for third-party apps.
    Authorization Code flow for web apps (with PKCE)
    Client Credentials flow for machine-to-machine

Authorization:
  RBAC (Role-Based Access Control): roles → permissions
  ABAC (Attribute-Based Access Control): policies based on attributes
  OPA (Open Policy Agent): policy-as-code for fine-grained authz

OWASP API Top 10 (2023):
  1. Broken Object Level Authorization (BOLA)
  2. Broken Authentication
  3. Broken Object Property Level Authorization
  4. Unrestricted Resource Consumption (rate limiting)
  5. Broken Function Level Authorization
  6. Server Side Request Forgery (SSRF)
  7. Security Misconfiguration
  8. Lack of Protection from Automated Threats
  9. Improper Inventory Management
  10. Unsafe Consumption of APIs

Best practices: validate all inputs, use HTTPS everywhere, rotate secrets,
monitor for anomalies, implement rate limiting, log access (not secrets).
""",
    },
]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Nexus Sample Data Ingestion ===")
    log.info("Connecting to Endee …")

    db  = EndeeClient()
    db.ensure_index(recreate=False)

    emb     = EmbeddingPipeline.get()
    pipeline = IngestionPipeline(db, emb)

    log.info("Ingesting %d documents …", len(DOCUMENTS))
    result = pipeline.ingest_documents(DOCUMENTS)

    log.info("✅ Done! Ingested %d documents → %d chunks",
             len(DOCUMENTS), result["total_chunks"])
    log.info("You can now start the backend and use the UI.")


if __name__ == "__main__":
    main()
