# ML Documentation Copilot

An AI-powered assistant for ML infrastructure documentation using advanced RAG techniques. Built with a modern monorepo architecture featuring Next.js frontend and FastAPI backend.

## 🏗️ Architecture

### Monorepo Structure

```
ml-docs-copilot/
├── packages/
│   ├── api/          # FastAPI backend
│   │   ├── api/      # API routes and logic
│   │   └── ingest/   # Data ingestion pipeline
│   └── web/          # Next.js frontend
│       └── src/
│           ├── app/        # Next.js app directory
│           ├── components/ # React components
│           ├── hooks/      # Custom React hooks
│           └── types/      # TypeScript types
├── package.json          # Root workspace configuration
└── pnpm-workspace.yaml   # Monorepo workspace config
```

## 🚀 Technology Stack

### Frontend
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4
- **UI Components**: Radix UI (headless components)
- **State Management**: TanStack Query v5
- **Markdown**: react-markdown with remark-gfm
- **Icons**: Lucide React
- **Language**: TypeScript 5+

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **LLM**: Google Gemini 2.5 Flash (Vertex AI)
- **Embeddings**: Vertex AI text-embedding-004 (768-dim)
- **Vector DB**: Qdrant Cloud (prod) / ChromaDB (dev)
- **Keyword Search**: SQLite FTS5 with BM25
- **Advanced Features**: Query expansion, MMR diversification, cross-encoder reranking

## ✨ Features

### Intelligent Query Routing
- **Smart Query Analyzer**: Client-side analysis to determine optimal backend endpoint
- **Tutorial Mode**: Structured step-by-step learning paths
- **Multi-Source Mode**: Cross-framework comparisons
- **Advanced Mode**: Full RAG pipeline with all optimizations
- **Standard Mode**: Fast basic queries

### Advanced RAG Pipeline
- **Hybrid Search**: Vector similarity + keyword search with RRF fusion
- **Query Expansion**: Multi-step breakdown for complex queries
- **Intent Classification**: Automatic query type detection
- **Vendor Detection**: Identifies ML frameworks mentioned
- **MMR Diversification**: Reduces result redundancy
- **Cross-Encoder Reranking**: Optimal relevance scoring
- **Context Expansion**: Fetches neighboring chunks for richer context

### Modern UI/UX
- **Interactive Tutorials**: Collapsible steps with navigation
- **Code Blocks**: Syntax highlighting with one-click copy
- **Visual Callouts**: Tips, warnings, info boxes
- **Citation Management**: Three-level citation system
- **Responsive Design**: Mobile-first, adaptive layout
- **Dark/Light Mode**: Theme support with smooth transitions

### Documentation Coverage
- PyTorch
- TensorFlow
- Scikit-learn
- MLflow
- Ray
- Weights & Biases

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and **pnpm**
- **Python** 3.11+
- **Google Cloud Platform** account with Vertex AI enabled
- **Qdrant Cloud** account (optional, for production)

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ml-docs-copilot

# Install frontend dependencies
pnpm install
```

### 2. Backend Setup

```bash
# Navigate to backend
cd packages/api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-api.txt

# Setup environment variables
cp env.example .env
# Edit .env with your credentials:
# - GOOGLE_PROJECT_ID
# - GOOGLE_LOCATION
# - GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
# - QDRANT_URL and QDRANT_API_KEY (if using Qdrant Cloud)
```

### 3. Data Ingestion (First Time)

```bash
# Run ingestion pipeline (takes 30-60 minutes)
cd packages/api
python -m ingest.main

# This will:
# - Crawl documentation from sources in ingest/seeds.yaml
# - Generate embeddings
# - Store in vector database and SQLite
```

### 4. Start Development Servers

```bash
# Terminal 1: Start backend (from packages/api)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend (from packages/web)
pnpm dev
```

### 5. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📡 API Endpoints

### Main Endpoints
- `POST /ask` - Standard Q&A (fast, basic RAG)
- `POST /ask-advanced` - Enhanced RAG with query expansion
- `POST /ask-multi-source` - Cross-framework queries
- `POST /howto` - Tutorial generation
- `GET /vendors` - Available documentation sources
- `GET /stats-comprehensive` - System statistics
- `GET /health` - Health check

## 🎨 Frontend Features

### Smart Query Routing
The frontend automatically analyzes queries and routes to the optimal endpoint:
- Tutorial mode toggle for step-by-step guides
- Automatic vendor detection (PyTorch, TensorFlow, etc.)
- Intent classification (how-to, conceptual, comparison)
- Query mode badges showing active features

### Enhanced Tutorial Interface
- **Floating Step Navigator**: Desktop sidebar with progress tracking
- **Collapsible Steps**: Clean, organized content hierarchy
- **Interactive Code Blocks**: Syntax highlighting + copy buttons
- **Visual Callouts**: Color-coded tips, warnings, info boxes
- **Smart Citations**: Hover tooltips and consolidated references
- **Prerequisite Checklists**: Interactive completion tracking
- **Mobile Responsive**: Adaptive layout for all screen sizes

## 📦 Deployment

### Frontend (Vercel)
1. Connect repository to Vercel
2. Set root directory to `packages/web`
3. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

### Backend (Railway)
1. Deploy from `packages/api` directory
2. Add environment variables:
   ```
   GOOGLE_PROJECT_ID=your-gcp-project
   GOOGLE_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your-api-key
   CORS_ORIGINS=https://your-frontend.vercel.app
   ```
3. Run ingestion to populate database

See `packages/api/RAILWAY_DEPLOYMENT.md` for detailed deployment guide.

## 🛠️ Development

### Project Commands

```bash
# Root level
pnpm install              # Install all dependencies

# Frontend (packages/web)
pnpm dev                  # Start dev server (localhost:3000)
pnpm build                # Production build
pnpm lint                 # ESLint check

# Backend (packages/api)
uvicorn api.main:app --reload  # Start dev server (localhost:8000)
python -m ingest.main          # Run data ingestion
```

### Adding Documentation Sources

Edit `packages/api/ingest/seeds.yaml`:

```yaml
sources:
  - name: new-framework
    url: https://new-framework.org/docs/
    enabled: true
```

Then rerun ingestion:
```bash
python -m ingest.main
```

## 🎯 Key Differentiators

- ✅ **No LangChain**: Custom RAG implementation for full control
- ✅ **Smart Frontend**: Client-side query analysis reduces backend load
- ✅ **Modern UI**: 10+ major UI/UX improvements
- ✅ **Production-Ready**: Docker, cloud deployment, comprehensive docs
- ✅ **Type-Safe**: Full TypeScript on frontend, Pydantic on backend

## 📊 Performance

| Metric | Value |
|--------|-------|
| Query Latency | 1-10s (mode-dependent) |
| Retrieval Precision | ~75-85% |
| Concurrent Users | 10-50 |
| Bundle Size | ~450KB gzipped |
| Storage | 500MB-2GB |

## 📚 Documentation

- **Frontend README**: `packages/web/README.md`
- **Backend README**: `packages/api/README.md`
- **Deployment Guide**: `packages/api/RAILWAY_DEPLOYMENT.md`
- **Production Considerations**: `packages/api/PRODUCTION_CONSIDERATIONS.md`
- **UI/UX Improvements**: `packages/web/UI_UX_IMPROVEMENTS.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test both frontend and backend
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

[Specify your license here]

## 🆘 Support

For issues and questions:
- Check the documentation in individual package READMEs
- Review API logs and error messages
- Open an issue with detailed reproduction steps

---

**Built with ❤️ using Next.js, FastAPI, and Google Vertex AI**
