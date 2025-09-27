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
```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (with defaults)
DATA_DIR=./data
CHROMA_COLLECTION=ml_docs
SQLITE_DB=bm25.db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_VECTOR=10
TOP_K_KEYWORD=10
RRF_K=60

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
