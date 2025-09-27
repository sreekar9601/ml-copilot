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

