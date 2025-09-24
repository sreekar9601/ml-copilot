# ML Documentation Copilot

An AI-powered assistant for ML infrastructure documentation, specializing in PyTorch, MLflow, Ray Serve, and KServe. The system uses a hybrid retrieval approach (semantic + keyword search) with strict citation requirements to provide accurate, source-backed answers.

## Features

- 🔍 **Hybrid Search**: Combines vector similarity and keyword search using Reciprocal Rank Fusion (RRF)
- 📚 **Comprehensive Knowledge Base**: Covers PyTorch, MLflow, KServe, and Ray Serve documentation
- 🎯 **Strict Citations**: Every factual statement is backed by source citations
- 🚀 **Fast API**: Built with FastAPI for high-performance REST endpoints
- 🐳 **Docker Ready**: Containerized for easy deployment
- ☁️ **Cloud Deploy**: Configured for Fly.io deployment with persistent storage

## Architecture

### Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Nomic AI nomic-embed-text-v1 (local)
- **Vector DB**: ChromaDB (persistent, file-based)
- **Keyword Search**: SQLite FTS5
- **Deployment**: Docker + Fly.io

### Data Flow

1. **Ingestion**: Crawl documentation → Clean HTML → Chunk text → Generate embeddings → Store in ChromaDB + SQLite
2. **Retrieval**: User query → Parallel vector + keyword search → RRF fusion → Context expansion
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

- `POST /ask` - Ask questions about ML documentation
  ```json
  {
    "q": "How to use MLflow model registry?",
    "top_k": 5,
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

### Fly.io Deployment

1. **Install Fly CLI**
   ```bash
   # Follow instructions at https://fly.io/docs/hands-on/install-flyctl/
   ```

2. **Deploy Application**
   ```bash
   # Launch app (don't deploy yet)
   fly launch --name ml-docs-copilot --no-deploy
   
   # Create persistent volume for data
   fly volumes create vectordata --size 3 --region sjc
   
   # Set API key secret
   fly secrets set GOOGLE_API_KEY="your_google_api_key_here"
   
   # Deploy
   fly deploy
   ```

3. **Initialize Data**
   ```bash
   # SSH into the deployed app and run ingestion
   fly ssh console
   python -m ingest.main --clear
   ```

### Production Considerations

- **Volume Size**: Start with 3GB, monitor usage and scale as needed
- **Memory**: 2GB RAM recommended for embedding model
- **Auto-scaling**: Configure based on expected traffic
- **Monitoring**: Use Fly.io metrics or add external monitoring
- **API Rate Limits**: Consider implementing rate limiting for production use

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
│   ├── prompts.py         # LLM prompts
│   └── config.py          # Configuration
├── ingest/                # Data ingestion pipeline
│   ├── main.py           # Ingestion runner
│   ├── crawl.py          # Web crawling
│   ├── chunker.py        # Text chunking
│   ├── embedder.py       # Text embedding
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

