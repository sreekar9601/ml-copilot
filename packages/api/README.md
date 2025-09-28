# ML Documentation Copilot

An AI-powered assistant for ML infrastructure documentation, specializing in PyTorch, MLflow, Ray Serve, and KServe. The system uses a **two-service architecture** with a lightweight API service and separate ingestion service for optimal performance.

## Features

- 🔍 **Advanced Hybrid Search**: Combines vector similarity and keyword search using Reciprocal Rank Fusion (RRF)
- 🧠 **Query Expansion**: Automatically generates multiple focused sub-queries for comprehensive retrieval
- 🎯 **Intelligent Re-ranking**: Cross-encoder re-ranking for optimal relevance scoring
- 📚 **Comprehensive Knowledge Base**: Covers PyTorch, MLflow, KServe, and Ray Serve documentation
- 🎯 **Strict Citations**: Every factual statement is backed by source citations
- 🚀 **Fast API**: Built with FastAPI for high-performance REST endpoints
- 🐳 **Docker Ready**: Containerized for easy deployment
- ☁️ **Cloud Deploy**: Railway + Qdrant Cloud deployment ready

## Architecture

### Two-Service Design

#### 🚀 API Service (Lightweight)
- **Backend**: FastAPI (Python 3.11)
- **LLM**: Google Gemini 1.5 Flash
- **Vector DB**: Qdrant Cloud (production) / ChromaDB (development)
- **Keyword Search**: SQLite FTS5
- **Deployment**: Railway with Docker
- **Dependencies**: Only essential libraries (FastAPI, ChromaDB, Google AI)

#### ⚙️ Ingestion Service (Heavy)
- **ML Libraries**: PyTorch, Transformers, Accelerate
- **Embeddings**: Nomic AI nomic-embed-text-v1 (ingestion) / Google Vertex AI (API)
- **Processing**: Document chunking, embedding generation
- **Usage**: Run manually when adding new documentation
- **Output**: Vector embeddings and search indices for API service

### Data Flow

1. **Ingestion**: Crawl documentation → Clean HTML → Chunk text → Generate embeddings → Store in ChromaDB + SQLite
2. **Advanced Retrieval**: 
   - User query → Query expansion → Multiple sub-queries
   - Parallel vector + keyword search for each sub-query
   - RRF fusion → Cross-encoder re-ranking → Context expansion
3. **Generation**: Formatted context + system prompt → Gemini API → Cited response

## Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API key
- Docker (for containerized deployment)
- Git

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd ml-docs-copilot
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Run Data Ingestion**
   ```bash
   python -m ingest.main --clear
   ```
   This will:
   - Crawl documentation from URLs in `ingest/seeds.yaml`
   - Process and chunk the content
   - Generate embeddings and store in databases
   - Take 10-15 minutes for initial run

4. **Start the API Server**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"q": "How to set up PyTorch DataLoader for distributed training?"}'
   ```

### Docker Development

1. **Build and Run**
   ```bash
   docker-compose up --build
   ```

2. **Run Ingestion Inside Container**
   ```bash
   docker-compose exec ml-docs-copilot python -m ingest.main --clear
   ```

## API Endpoints

### Core Endpoints

- `POST /ask` - Ask questions about ML documentation (standard retrieval)
  ```json
  {
    "q": "How to use MLflow model registry?",
    "top_k": 5,
    "include_sources": true
  }
  ```

- `POST /ask-advanced` - Ask questions with advanced query expansion and re-ranking
  ```json
  {
    "q": "Design a machine learning system for fraud detection",
    "top_k": 10,
    "include_sources": true
  }
  ```

- `GET /health` - Health check and system status
- `GET /stats` - Knowledge base statistics
- `GET /sources/{chunk_id}` - Get detailed chunk information

### Admin Endpoints

- `POST /reindex` - Trigger reindexing (background task)
  ```json
  {
    "clear_existing": true
  }
  ```

## Deployment

### Railway + Qdrant Cloud (Recommended)

For production deployment, see **[RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)** for complete setup guide.

**Quick Setup:**
1. **Create Qdrant Cloud cluster** (free tier: 1GB)
2. **Get Google Gemini API key** from [ai.google.dev](https://ai.google.dev)
3. **Deploy to Railway** with environment variables
4. **Run ingestion** to populate Qdrant with documentation

### Local Development

1. **Copy environment template**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

2. **Run ingestion locally**
   ```bash
   python ingest/ingest_focused_web.py
   ```

3. **Start API server**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

### Production Considerations

- **Qdrant Storage**: Free tier provides 1GB (330k vectors)
- **Railway Scaling**: Auto-scales based on traffic
- **API Rate Limits**: Consider implementing for public deployment
- **Monitoring**: Use Railway metrics and Qdrant Cloud dashboard
- **Security**: Store API keys securely in Railway environment variables

## Configuration

### Environment Variables

All configuration is handled through environment variables:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (with defaults)
DATA_DIR=./data
CHROMA_COLLECTION=ml_docs
SQLITE_DB=bm25.db
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_VECTOR=10
TOP_K_KEYWORD=10
RRF_K=60
```

### Adding New Documentation Sources

Edit `ingest/seeds.yaml` to add new documentation URLs:

```yaml
urls:
  - https://new-ml-framework.org/docs/guide
  - https://another-tool.io/documentation
```

Then rerun ingestion:
```bash
python -m ingest.main --clear
```

## Advanced Retrieval System

### Query Expansion

The system automatically expands user queries into multiple focused sub-queries:

- **Architectural Queries**: For system design questions, generates sub-queries about best practices, deployment patterns, and architectural patterns
- **Implementation Queries**: For "how-to" questions, generates sub-queries about examples, API references, and step-by-step guides
- **Comparison Queries**: For technology comparisons, generates sub-queries comparing different tools and approaches
- **Best Practices**: Always generates sub-queries about best practices and common patterns
- **Troubleshooting**: Generates sub-queries about common issues and debugging

### Re-ranking System

After initial retrieval, the system uses intelligent re-ranking:

- **Term Overlap Scoring**: Boosts results with higher query-term overlap
- **Title Relevance**: Prioritizes results where titles match query terms
- **Content Relevance**: Analyzes content similarity to the original query
- **Cross-encoder Scoring**: Uses advanced scoring for optimal relevance

### Retrieval Modes

- **Standard Mode** (`/ask`): Uses basic hybrid search with RRF fusion
- **Advanced Mode** (`/ask-advanced`): Uses query expansion + re-ranking for complex architectural questions

## Performance Tuning

### Retrieval Parameters

- `TOP_K_VECTOR`: Number of vector search results (default: 10)
- `TOP_K_KEYWORD`: Number of keyword search results (default: 10)
- `RRF_K`: RRF parameter for fusion (default: 60, higher = more conservative)

### Chunking Parameters

- `CHUNK_SIZE`: Target chunk size in tokens (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)

### Model Selection

The system uses `nomic-ai/nomic-embed-text-v1` by default for its balance of performance and quality. For production, consider:

- **Higher quality**: Larger embedding models (but slower)
- **Faster inference**: Smaller models or quantized versions
- **Cost optimization**: Cached embeddings, batch processing

## Troubleshooting

### Common Issues

1. **Empty retrieval results**
   - Check if ingestion completed successfully
   - Verify databases exist in `data/` directory
   - Check API logs for errors

2. **Slow response times**
   - First request loads embedding model (slower)
   - Consider pre-warming with health checks
   - Monitor resource usage

3. **Memory issues**
   - Embedding model requires ~1GB RAM
   - Increase container memory allocation
   - Use CPU-only torch installation

4. **Deployment failures**
   - Verify API key is set correctly
   - Check volume mounting
   - Review Fly.io logs: `fly logs`

### Debugging

```bash
# Check API logs
docker-compose logs ml-docs-copilot

# Fly.io logs
fly logs

# Test retrieval directly
python -c "from api.retrieval import retrieve_documents; print(retrieve_documents('test query'))"

# Database stats
curl http://localhost:8000/stats
```

## Development

### Project Structure

```
ml-docs-copilot/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── retrieval.py       # Hybrid search system
│   ├── advanced_retrieval.py # Advanced retrieval with expansion & re-ranking
│   ├── clients.py         # Google AI client configuration
│   ├── prompts.py         # LLM prompts
│   └── config.py          # Configuration
├── ingest/                # Data ingestion pipeline
│   ├── main.py           # Ingestion runner
│   ├── crawl.py          # Web crawling
│   ├── chunker.py        # Text chunking
│   ├── embedder.py        # Text embedding
│   ├── upsert.py         # Database storage
│   └── seeds.yaml        # Documentation URLs
├── data/                 # Persistent databases
├── docker/               # Docker configuration
└── fly.toml             # Fly.io configuration
```

### Adding Features

1. **New documentation sources**: Add URLs to `seeds.yaml`
2. **Enhanced retrieval**: Modify `api/retrieval.py`
3. **Better prompts**: Update `api/prompts.py`
4. **New endpoints**: Extend `api/main.py`

### Testing

```bash
# Run ingestion test
python -m ingest.main --test

# Test retrieval
python -m api.retrieval

# API testing
pytest tests/  # If test suite is added
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review API logs and error messages
- Open an issue with reproduction steps

