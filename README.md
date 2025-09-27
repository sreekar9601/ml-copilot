<<<<<<< HEAD
# ML Documentation Copilot - Full Stack

A complete full-stack application for AI-powered assistance with ML infrastructure documentation. Built with a modern monorepo architecture featuring a Next.js frontend and FastAPI backend.

## 🏗️ Architecture

### Monorepo Structure

```
ml-copilot-monorepo/
├── packages/
│   ├── api/          # FastAPI backend
│   │   ├── api/      # API routes and logic
│   │   ├── ingest/   # Data ingestion pipeline
│   │   └── data/     # Local databases
│   └── web/          # Next.js frontend
│       ├── src/
│       │   ├── app/        # Next.js app directory
│       │   ├── components/ # React components
│       │   ├── hooks/      # Custom React hooks
│       │   └── types/      # TypeScript types
│       └── public/   # Static assets
├── package.json      # Root workspace configuration
└── pnpm-workspace.yaml
```

### Technology Stack

#### Frontend (packages/web)
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui
- **State Management**: TanStack Query (React Query)
- **Markdown Rendering**: react-markdown with remark-gfm
- **Icons**: Lucide React
- **Language**: TypeScript

#### Backend (packages/api)
- **Framework**: FastAPI (Python 3.11+)
- **LLM**: Google Gemini 1.5 Flash
- **Vector DB**: Qdrant Cloud (production) / ChromaDB (development)
- **Search**: Hybrid search with RRF (Reciprocal Rank Fusion)
- **Embeddings**: Nomic AI embeddings

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and **pnpm** (for frontend)
- **Python** 3.11+ (for backend)
- **Google Gemini API key**
- **Qdrant Cloud account** (optional, for production)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ml-copilot-monorepo
pnpm install
```

### 2. Backend Setup

```bash
# Navigate to backend
cd packages/api

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Ingest Documentation (First Time Only)

```bash
# Run data ingestion (takes 10-15 minutes)
python -m ingest.main --clear
```

### 4. Start Development Servers

From the monorepo root:

```bash
# Start both frontend and backend concurrently
pnpm dev

# Or start individually:
pnpm dev:web    # Frontend only (http://localhost:3000)
pnpm dev:api    # Backend only (http://localhost:8000)
```

### 5. Open Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 Features

### Chat Interface
- Modern, responsive chat UI
- Real-time message streaming
- Markdown rendering for AI responses
- Auto-scrolling message history
- Loading states and error handling

### AI Capabilities
- **Hybrid Search**: Vector similarity + keyword search
- **Cited Responses**: Every answer includes source citations
- **Specialized Knowledge**: PyTorch, MLflow, KServe, Ray Serve
- **Context-Aware**: Maintains conversation context

### Developer Experience
- **Type Safety**: Full TypeScript support
- **Component Library**: Consistent UI with shadcn/ui
- **Hot Reload**: Fast development with Next.js and FastAPI
- **Error Boundaries**: Graceful error handling

## 📦 Deployment

### Frontend (Vercel)

1. **Connect Repository**
   - Deploy to Vercel
   - Set root directory to `packages/web`

2. **Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

### Backend (Railway)

1. **Deploy from `packages/api` directory**

2. **Environment Variables**
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   QDRANT_URL=https://your-cluster.qdrant.tech
   QDRANT_API_KEY=your_qdrant_api_key
   ALLOWED_ORIGINS=https://your-frontend-url.vercel.app
   ```

3. **Run Ingestion**
   ```bash
   # SSH into Railway container or run locally
   python -m ingest.main --clear
   ```

## 🛠️ Development

### Adding New Features

1. **Frontend Components**
   ```bash
   cd packages/web
   # Add new shadcn/ui components
   npx shadcn@latest add <component-name>
   ```

2. **Backend Endpoints**
   - Add routes in `packages/api/api/main.py`
   - Update types in `packages/web/src/types/`

### Available Scripts

```bash
# Root level commands
pnpm dev          # Start both frontend and backend
pnpm dev:web      # Start frontend only
pnpm dev:api      # Start backend only
pnpm build        # Build frontend for production
pnpm lint         # Lint frontend code

# Frontend specific (packages/web)
pnpm dev          # Development server
pnpm build        # Production build
pnpm start        # Start production server
pnpm lint         # ESLint

# Backend specific (packages/api)
python -m uvicorn api.main:app --reload  # Development server
python -m ingest.main --clear             # Re-ingest documentation
```

### Environment Variables

#### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### Backend (.env)
=======
# ML Documentation Copilot

An AI-powered assistant for ML infrastructure documentation, specializing in PyTorch, MLflow, Ray Serve, and KServe. The system uses a **two-service architecture** with a lightweight API service and separate ingestion service for optimal performance.

## Features

- 🔍 **Hybrid Search**: Combines vector similarity and keyword search using Reciprocal Rank Fusion (RRF)
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

>>>>>>> f01a73dc8acea0b2c61a76e10ebcd70be48c0603
```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (with defaults)
DATA_DIR=./data
CHROMA_COLLECTION=ml_docs
SQLITE_DB=bm25.db
<<<<<<< HEAD
=======
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
>>>>>>> f01a73dc8acea0b2c61a76e10ebcd70be48c0603
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_VECTOR=10
TOP_K_KEYWORD=10
RRF_K=60
<<<<<<< HEAD

# Production
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

## 📝 API Documentation

The backend automatically generates OpenAPI documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /ask` - Submit questions to the AI
- `GET /health` - Health check
- `GET /stats` - Knowledge base statistics
- `GET /sources/{chunk_id}` - Get source details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request

## 📄 License

[Specify your license here]

## 🆘 Support

For issues and questions:
- Check the troubleshooting section in individual package READMEs
- Review API logs and error messages
- Open an issue with reproduction steps

---

**Built with ❤️ using Next.js, FastAPI, and Google Gemini**
=======
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

>>>>>>> f01a73dc8acea0b2c61a76e10ebcd70be48c0603
