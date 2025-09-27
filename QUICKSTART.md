<<<<<<< HEAD
# 🚀 Quick Start Guide

## What We've Built

A complete full-stack RAG application with:

### ✅ Frontend (Next.js)
- **Modern Chat Interface** with real-time messaging
- **Beautiful UI** using shadcn/ui components
- **Markdown Rendering** for AI responses with syntax highlighting
- **TypeScript** for type safety
- **Responsive Design** that works on all devices

### ✅ Backend (FastAPI)
- **Hybrid Search** combining vector similarity and keyword search
- **Google Gemini Integration** for AI responses
- **CORS Configuration** for frontend communication
- **Production Ready** with Railway deployment support

### ✅ Monorepo Architecture
- **Single Repository** for both frontend and backend
- **pnpm Workspaces** for efficient dependency management
- **Concurrent Development** with `pnpm dev`

## 🏃‍♂️ Running the Application

### 1. Start Both Services
```bash
# From the monorepo root
pnpm dev
```

This will start:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000

### 2. Backend Setup (First Time Only)

```bash
# Navigate to backend
cd packages/api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run data ingestion (takes 10-15 minutes)
python -m ingest.main --clear
```

### 3. Test the Application

1. **Open Frontend**: http://localhost:3000
2. **Try asking**: "How to set up distributed training with PyTorch?"
3. **Check API**: http://localhost:8000/docs

## 🎯 Key Features

### Chat Interface
- **Real-time messaging** with loading states
- **Auto-scrolling** message history
- **Error handling** with user-friendly messages
- **Markdown rendering** for code blocks and formatting

### AI Capabilities
- **Specialized knowledge** in PyTorch, MLflow, KServe, Ray Serve
- **Source citations** for every response
- **Context-aware** responses
- **Hybrid search** for better accuracy

### Developer Experience
- **Type safety** throughout the stack
- **Hot reload** for fast development
- **Component library** for consistent UI
- **Error boundaries** for graceful failures

## 🚀 Deployment

### Frontend (Vercel)
1. Connect repository to Vercel
2. Set root directory to `packages/web`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app`

### Backend (Railway)
1. Deploy from `packages/api` directory
2. Add environment variables:
   - `GOOGLE_API_KEY=your_gemini_api_key`
   - `QDRANT_URL=https://your-cluster.qdrant.tech`
   - `QDRANT_API_KEY=your_qdrant_api_key`
   - `ALLOWED_ORIGINS=https://your-frontend.vercel.app`
3. Run ingestion: `python -m ingest.main --clear`

## 📁 Project Structure

```
ml-copilot-monorepo/
├── packages/
│   ├── api/              # FastAPI backend
│   │   ├── api/          # API routes
│   │   ├── ingest/       # Data pipeline
│   │   └── data/         # Local databases
│   └── web/              # Next.js frontend
│       ├── src/
│       │   ├── app/      # Next.js app directory
│       │   ├── components/chat/  # Chat components
│       │   ├── hooks/    # API hooks
│       │   └── types/    # TypeScript types
│       └── public/       # Static assets
├── package.json          # Root workspace config
└── pnpm-workspace.yaml
```

## 🛠️ Development Commands

```bash
# Root level
pnpm dev          # Start both services
pnpm dev:web      # Frontend only
pnpm dev:api      # Backend only

# Frontend (packages/web)
pnpm dev          # Development server
pnpm build        # Production build
pnpm start        # Production server

# Backend (packages/api)
python -m uvicorn api.main:app --reload  # Development server
python -m ingest.main --clear            # Re-ingest data
```

## 🎉 Success!

You now have a complete, production-ready RAG application with:

- ✅ Modern React frontend with TypeScript
- ✅ FastAPI backend with AI integration
- ✅ Beautiful, responsive chat interface
- ✅ Monorepo architecture for easy development
- ✅ Ready for deployment to Vercel + Railway

**Happy coding! 🚀**
=======
# ML Documentation Copilot - Quick Start Guide

## ✅ Installation Status
Your installation is working correctly! Key components verified:
- ✅ All Python dependencies installed
- ✅ Embedding model (Nomic AI) downloaded and working
- ✅ FastAPI application ready
- ✅ Database systems (ChromaDB + SQLite) functional

## 🚀 Next Steps

### 1. Set up your API key
Create a `.env` file with your Google Gemini API key:

```bash
# Copy the example file
cp env.example .env

# Edit .env and add your API key:
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 2. Ingest the documentation
This will crawl and process ML documentation from PyTorch, MLflow, KServe, and Ray Serve:

```bash
python run.py ingest --clear
```

**Note:** This will take 10-15 minutes and download ~2GB of documentation. It's a one-time setup.

### 3. Start the API server
```bash
python run.py start-api --reload
```

The server will start at: http://localhost:8000

### 4. Test the system
Open your browser to http://localhost:8000/docs for the interactive API documentation, or test with curl:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q": "How to set up PyTorch DataLoader for distributed training?"}'
```

## 📚 Example Questions to Try

Once your system is running, try these questions:

- "How to configure PyTorch DistributedDataParallel?"
- "What are MLflow model registry best practices?"
- "How to deploy a KServe InferenceService?"
- "What are Ray Serve autoscaling options?"

## 🔧 Troubleshooting

### If ingestion fails:
- Check your internet connection
- Verify the `data/` directory is writable
- Check logs for specific errors

### If API requests fail:
- Verify your GOOGLE_API_KEY is set correctly
- Check the `/health` endpoint: http://localhost:8000/health
- Review API logs for errors

### Performance tips:
- First request may be slower (model loading)
- Subsequent requests will be much faster
- Consider pre-warming with a test query

## 🎯 What's Working

Your system includes:
- **Hybrid Search**: Vector similarity + keyword matching
- **Strict Citations**: Every answer includes source references
- **Context Expansion**: Automatic inclusion of surrounding content
- **Production Ready**: Docker and cloud deployment configs included

## 📊 System Status

Check system statistics at any time:
```bash
curl http://localhost:8000/stats
```

This shows:
- Number of documents indexed
- Source distribution
- Database health

---

**You're all set!** The ML Documentation Copilot is ready to help with your ML infrastructure questions.

>>>>>>> f01a73dc8acea0b2c61a76e10ebcd70be48c0603
