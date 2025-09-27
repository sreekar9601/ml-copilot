Of course. Here is a detailed architecture and execution plan for building a sophisticated, Retrieval-Augmented Generation (RAG) application. This plan is designed to be directly fed to an AI coding assistant.

-----

## **Project: ML Documentation Copilot**

### **1. Overview & Goals**

The primary goal is to build an AI assistant that answers questions about complex machine learning pipelines (PyTorch, MLflow, Ray Serve, KServe). The assistant must provide step-by-step plans, include relevant code snippets, and strictly cite its sources from the provided documentation.

  * **Core Function:** Answer user queries based on a curated set of technical documentation.
  * **Key Feature 1:** Implement a **hybrid search** retrieval system (semantic + keyword) for accuracy.
  * **Key Feature 2:** Enforce **strict, inline citations** for every factual statement in the generated answer to ensure verifiability and combat hallucination.
  * **Key Feature 3:** Package the system as a **FastAPI service** and deploy it on a free-tier cloud platform.

-----

### **2. Technology Stack**

  * **Backend Framework:** **FastAPI** (Python 3.11)
  * **LLM:** **Google Gemini 1.5 Flash** (via `google-generativeai` SDK)
  * **Embedding Model:** **`nomic-ai/nomic-embed-text-v1`** (run locally via `transformers`)
  * **Vector Database:** **ChromaDB** (persistent, file-based)
  * **Keyword Search:** **SQLite FTS5** (built-in, high-performance full-text search)
  * **Data Ingestion:** `httpx`, `BeautifulSoup4`, `readability-lxml`, `markdownify`
  * **Deployment:** **Docker** & **Fly.io** (with a persistent volume)
  * **Observability (Optional):** **Langfuse** for tracing and debugging.

-----

### **3. Directory Structure**

Create the following file and directory structure for the project.

```
ml-docs-copilot/
├── api/
│   ├── main.py             # FastAPI app (/ask endpoint)
│   ├── retrieval.py        # Hybrid search logic (Chroma + SQLite + RRF)
│   ├── prompts.py          # System and self-check prompts
│   └── config.py           # Pydantic settings for env vars
├── ingest/
│   ├── main.py             # Main ingestion script runner
│   ├── seeds.yaml          # List of canonical doc URLs to crawl
│   ├── crawl.py            # Fetches and cleans HTML from URLs
│   ├── chunker.py          # Splits Markdown into semantic chunks
│   ├── embedder.py         # Handles text embedding with Nomic
│   └── upsert.py           # Inserts chunks into ChromaDB & SQLite
├── data/                   # (Will be created) For persistent databases
├── docker/
│   └── Dockerfile.api      # Dockerfile for the API service
├── .env.example            # Template for environment variables
├── .gitignore
├── docker-compose.yml      # For local development
├── fly.toml                # Fly.io deployment configuration
└── requirements.txt
```

-----

### **4. Data Sources & Ingestion Pipeline**

The knowledge base will be built from official documentation.

#### **`ingest/seeds.yaml`**

```yaml
urls:
  # PyTorch
  - https://pytorch.org/docs/stable/data.html
  - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
  - https://pytorch.org/docs/stable/notes/ddp.html
  # MLflow
  - https://mlflow.org/docs/latest/concepts.html
  - https://mlflow.org/docs/latest/model-registry.html
  # KServe
  - https://kserve.github.io/website/0.11/get_started/
  - https://kserve.github.io/website/0.11/modelserving/inferenceservice/
  # Ray Serve
  - https://docs.ray.io/en/latest/serve/key-concepts.html
  - https://docs.ray.io/en/latest/serve/production-guide/config-file.html
```

#### **Ingestion Flow**

The ingestion process is an offline script that populates the databases.

1.  **Crawl:** `ingest/crawl.py` will read URLs from `seeds.yaml`, fetch the HTML using `httpx`, and parse the main article content using `readability-lxml`. It will also add `id` tags to headings for deep linking.
2.  **Chunk:** `ingest/chunker.py` will take the cleaned Markdown and split it into semantic chunks of about 500 tokens. Each chunk will store its text, source URL, heading path (`Docs > Section > Subsection`), and anchor link. Crucially, it will also link to the previous and next chunks (`prev_id` and `next_id`) to enable context expansion.
3.  **Embed:** `ingest/embedder.py` will initialize the `nomic-embed-text-v1` model and provide a method to encode a batch of text chunks into vector embeddings.
4.  **Upsert:** `ingest/upsert.py` will orchestrate the final step. It takes the chunks, generates embeddings for them, and then inserts the data into two places:
      * **ChromaDB:** Stores `chunk_id`, vector embedding, and full metadata.
      * **SQLite FTS5:** Stores `chunk_id` and the text content for efficient keyword search.

-----

### **5. API & Retrieval-Augmented Generation Flow**

The API serves real-time requests from users.

#### **`api/retrieval.py`**

This is the core of the RAG system.

1.  **Receive Query:** A user query (e.g., "How to set up DDP?") is received.
2.  **Parallel Search:**
      * **Vector Search:** The query is embedded using the Nomic model, and ChromaDB is queried to find the top-K semantically similar chunks.
      * **Keyword Search:** The raw query is used to search the SQLite FTS5 table for chunks containing the exact keywords.
3.  **Fuse Results (RRF):** The ranked lists from both searches are combined using **Reciprocal Rank Fusion (RRF)**. This algorithm prioritizes items that rank highly in both lists, providing a robust hybrid score without needing to tune weights.
4.  **Expand Context:** The top-N results from the fused list are selected. For each result, we also fetch its preceding and succeeding chunks (using `prev_id` and `next_id`) to provide the LLM with a more complete context.
5.  **Return Context:** The final, expanded list of chunk texts and their metadata is returned.

#### **`api/main.py`**

This file defines the FastAPI endpoint.

1.  **`/ask` Endpoint:** Receives a request containing the user's query `q`.
2.  **Retrieve Context:** Calls the `retrieve` function from `retrieval.py` to get the relevant document chunks.
3.  **Construct Prompt:** Formats the retrieved chunks and the user query into a detailed prompt for Gemini, using the system prompt from `prompts.py`. The prompt strictly instructs the model to only use the provided context and to cite every sentence.
4.  **Generate Answer:** Sends the prompt to the Gemini API.
5.  **Post-process & Return:** The LLM's response is lightly processed (e.g., to ensure citation format) and returned to the user in a JSON object containing the answer, the list of sources used, and latency metrics.

-----

### **6. Deployment Plan (Fly.io)**

#### **`requirements.txt`**

```
fastapi uvicorn pydantic python-dotenv
chromadb
sqlite-utils
numpy
transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cpu
beautifulsoup4 readability-lxml markdownify html5lib
google-generativeai
```

#### **`docker/Dockerfile.api`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Use CPU-only torch wheels for smaller image size
RUN pip install --no-cache-dir -r requirements.txt

ENV DATA_DIR=/app/data
RUN mkdir -p /app/data

COPY ./api /app/api
COPY ./ingest /app/ingest

# Pre-download the embedding model during the build
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    model_id='nomic-ai/nomic-embed-text-v1'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id)"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **`fly.toml`**

```toml
app = "ml-docs-copilot"
primary_region = "sjc" # Choose a region near you

[build]
  dockerfile = "docker/Dockerfile.api"

# Mount a persistent volume at /app/data to store the Chroma/SQLite databases
[mounts]
  source="vectordata"
  destination="/app/data"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0 # Set to 1 for faster responses
```

#### **Deployment Commands**

```bash
# 1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/

# 2. Launch the app (don't deploy yet)
fly launch --name ml-docs-copilot --no-deploy

# 3. Create a persistent volume for the data (3GB is a good start)
fly volumes create vectordata --size 3 --region <your-chosen-region>

# 4. Set secrets (your API key)
fly secrets set GOOGLE_API_KEY="your_google_api_key_here"

# 5. Deploy the application
fly deploy
```

-----

### **7. Execution Plan**

#### **Day 1: Ingestion & Core API**

1.  **Setup:** Initialize the project structure, set up the virtual environment, and install dependencies from `requirements.txt`.
2.  **Ingestion:** Implement the full ingestion pipeline (`crawl` -\> `chunk` -\> `embed` -\> `upsert`). Run it locally to populate the `data/` directory with `chroma.sqlite3` and `bm25.db`.
3.  **Retrieval:** Implement the `retrieval.py` module with vector search, keyword search, and RRF.
4.  **API:** Build the `/ask` endpoint in `api/main.py`. Test it locally by sending `curl` requests.
5.  **Initial Deploy:** Create the `Dockerfile` and `fly.toml`, and deploy the first version to Fly.io.

#### **Day 2: Refinement, Evaluation & Ops**

1.  **Evaluation:** Create a small test suite of 10-20 questions. Write a script to measure retrieval recall (did you find the right documents?) and use an LLM-as-judge pattern to evaluate answer quality and groundedness.
2.  **Prompt Tuning:** Based on evaluation results, refine the system prompt in `prompts.py` to improve answer formatting and citation strictness.
3.  **Add Re-indexing Endpoint:** Create a secure `/reindex` endpoint on the API that can be triggered to re-run the ingestion pipeline on the deployed machine.
4.  **Documentation:** Write a `README.md` explaining how to set up the project, run it locally, and use the API.
5.  **Frontend (Optional):** Build a simple Streamlit or Next.js interface that provides a UI for the copilot.