# ML Documentation Copilot

> AI-powered assistant for navigating ML framework documentation using advanced RAG and agentic orchestration.

**Query across PyTorch, TensorFlow, Scikit-learn, MLflow, Ray, and W&B docs in natural language.**

---

## 🎯 What It Does

- **Smart Q&A**: Ask questions, get accurate answers with citations
- **Tutorial Generation**: Get step-by-step guides for any ML task
- **Cross-Framework Comparison**: Compare approaches across libraries
- **Agentic Tools** (v3.0): Web search, code execution, adaptive reasoning

---

## 🏗️ Architecture

```
ml-docs-copilot/
├── packages/
│   ├── api/        # FastAPI backend (Python 3.11+)
│   │   ├── api/    # RAG pipeline & endpoints
│   │   ├── agent/  # LangGraph orchestration (v3.0)
│   │   └── ingest/ # Data ingestion pipeline
│   └── web/        # Next.js 15 frontend (React 19)
└── pnpm-workspace.yaml
```

**Tech Stack**: Next.js 15 • FastAPI • Google Gemini 2.5 Flash • Qdrant • LangGraph • Tailwind CSS v4

---

## ⚡ Quick Start

### 1. Prerequisites

- Node.js 18+, pnpm 8+
- Python 3.11+
- Google Cloud Platform account (Vertex AI enabled)

### 2. Install Dependencies

```bash
# Root: Install frontend
pnpm install

# Backend: Create venv and install
cd packages/api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-api.txt
```

### 3. Configure Environment

```bash
cd packages/api
cp env.example .env
```

**Edit `.env` with:**
```bash
# Required
GOOGLE_PROJECT_ID=your-gcp-project
GOOGLE_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-credentials.json

# Optional (graceful fallback if missing)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
TAVILY_API_KEY=your-tavily-key      # Web search
E2B_API_KEY=your-e2b-key            # Code execution
```

### 4. Ingest Documentation (First Time)

```bash
# From packages/api (takes 30-60 min)
python -m ingest.main
```

### 5. Start Development Servers

```bash
# Terminal 1: Backend (from packages/api)
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend (from packages/web)
pnpm dev
```

**Access**: http://localhost:3000

---

## 🚀 Features

### Intelligent Query Modes

| Mode | Best For | Latency |
|------|----------|---------|
| **Standard** | Quick factual queries | 1-2s |
| **Advanced** | Complex questions with query expansion | 2-4s |
| **Tutorial** | Step-by-step how-to guides | 4-8s |
| **Multi-Source** | Cross-framework comparisons | 3-5s |
| **Agent** (v3.0) | Tool-using, reasoning tasks | 2-12s |

### Advanced RAG Pipeline

- **Hybrid Search**: Vector similarity + BM25 keyword search with RRF fusion
- **Query Expansion**: Multi-step breakdown for complex queries
- **MMR Diversification**: Reduces result redundancy
- **Cross-Encoder Reranking**: Optimal relevance scoring
- **Context Expansion**: Fetches neighboring chunks

### Agentic System (v3.0 - Beta)

Powered by LangGraph with:
- **Tools**: Documentation search, web search (Tavily), code execution (E2B)
- **Memory**: SQLite-backed conversation history with cost tracking
- **Budget Enforcement**: Per-session limits (~$0.50 default)
- **Checkpointing**: Persistent state across sessions

---

## 📡 API Endpoints

### Standard RAG
```
POST   /ask                    # Basic Q&A
POST   /ask-advanced           # Enhanced RAG
POST   /ask-multi-source       # Cross-framework
POST   /howto                  # Tutorial generation
GET    /vendors                # Available frameworks
GET    /stats-comprehensive    # System stats
GET    /health                 # Health check
```

### Agent System (v3.0)
```
POST   /agent/ask              # Agentic query with tools
POST   /agent/stream           # SSE streaming
GET    /agent/conversations/{id}
DELETE /agent/conversations/{id}
```

**API Docs**: http://localhost:8000/docs

---

## 🎨 Frontend Features

- **Smart Query Routing**: Client-side analysis picks optimal backend mode
- **Interactive Tutorials**: Collapsible steps with floating navigator
- **Code Blocks**: Syntax highlighting with one-click copy
- **Citations**: Three-level system with hover tooltips
- **Dark/Light Mode**: Smooth theme transitions
- **Mobile-First**: Fully responsive design

---

## 📦 Deployment

### Frontend → Vercel

1. Connect repo to Vercel
2. Set root directory: `packages/web`
3. Add env var: `NEXT_PUBLIC_API_URL=https://your-backend.railway.app`

### Backend → Railway

1. Deploy from `packages/api`
2. Add environment variables (see `.env.example`)
3. Run `python -m ingest.main` post-deploy to populate database

**See**: `packages/api/RAILWAY_DEPLOYMENT.md` for detailed guide.

---

## 🛠️ Development

### Project Commands

```bash
# Root
pnpm install              # Install all deps

# Frontend (packages/web)
pnpm dev                  # Dev server (localhost:3000)
pnpm build                # Production build
pnpm lint                 # ESLint

# Backend (packages/api)
uvicorn api.main:app --reload    # Dev server (localhost:8000)
python -m ingest.main            # Run ingestion
python run_tests.py              # Run test suite
```

### Adding Documentation Sources

Edit `packages/api/ingest/seeds.yaml`:

```yaml
sources:
  - name: new-framework
    url: https://new-framework.org/docs/
    enabled: true
```

Then run: `python -m ingest.main`

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Query Latency (Standard) | 1-2s |
| Query Latency (Agent) | 2-12s |
| Retrieval Precision | 75-85% |
| Cost per Query | $0.001-$0.01 |
| Bundle Size | ~450KB gzipped |
| Concurrent Users | 10-50 |
| Storage | 500MB-2GB |

---

## 🗺️ Project Status

### ✅ Production-Ready
- Multi-mode RAG system with hybrid search
- Modern Next.js frontend with 10+ UI/UX improvements
- Tutorial generation with step-by-step guides
- Cross-framework comparisons
- Docker support

### 🚧 Beta (Agent v3.0)
- LangGraph orchestration (core complete)
- Tool integration (doc search, web, code execution)
- Conversation memory & cost tracking
- State checkpointing
- API endpoints & SSE streaming

### 📋 Planned
- Specialized agents (debug, tutorial)
- Self-reflection for quality
- Comprehensive evaluation suite
- Multi-agent collaboration

---

## 📚 Documentation

- **Agent System**: [`packages/api/agent/README.md`](packages/api/agent/README.md)
- **Frontend Guide**: [`packages/web/README.md`](packages/web/README.md)
- **Deployment**: [`packages/api/RAILWAY_DEPLOYMENT.md`](packages/api/RAILWAY_DEPLOYMENT.md)
- **Testing**: [`packages/api/TESTING_QUICK_REFERENCE.md`](packages/api/TESTING_QUICK_REFERENCE.md)
- **API Usage**: [`packages/api/API_USAGE_GUIDE.md`](packages/api/API_USAGE_GUIDE.md)

---

## 🎯 Key Differentiators

| Feature | Benefit |
|---------|---------|
| **No LangChain for RAG** | Full control, custom optimizations |
| **Client-Side Intelligence** | Reduced backend load |
| **Hybrid Architecture** | Classic RAG + agentic capabilities |
| **Production UX** | Tutorial mode, code highlighting, responsive |
| **Cost-Aware** | Built-in budget tracking |
| **Type-Safe** | TypeScript + Pydantic |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open Pull Request

---

## 📄 License

[Specify your license]

---

## 🆘 Troubleshooting

<details>
<summary><strong>ModuleNotFoundError: No module named 'langchain_google_vertexai'</strong></summary>

```bash
pip install -r requirements-api.txt
```
</details>

<details>
<summary><strong>Web search unavailable</strong></summary>

Set `TAVILY_API_KEY` in `.env` or disable with `AGENT_ENABLE_WEB_SEARCH=false`
</details>

<details>
<summary><strong>Budget exceeded error</strong></summary>

Increase limit: `AGENT_MAX_COST_PER_SESSION=1.0` or start new conversation
</details>

<details>
<summary><strong>CORS errors in frontend</strong></summary>

Set `CORS_ORIGINS` in backend `.env` to match your frontend URL
</details>

---

**Built with ❤️ using Next.js, FastAPI, LangGraph, and Google Vertex AI**
