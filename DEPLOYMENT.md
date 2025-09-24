# ML Documentation Copilot - Deployment Guide

## Railway Deployment

### Prerequisites
1. Railway account (free tier available)
2. GitHub repository with your code
3. Google API key for Gemini

### Quick Deploy to Railway

1. **Connect to Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign up/login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Set Environment Variables:**
   In Railway dashboard, go to Variables tab and add:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   DATA_DIR=/app/data
   CHROMA_COLLECTION=ml_docs
   SQLITE_DB=bm25.db
   EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   TOP_K_VECTOR=10
   TOP_K_KEYWORD=10
   RRF_K=60
   ```

3. **Deploy:**
   - Railway will automatically build and deploy
   - The app will be available at `https://your-app-name.railway.app`

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

3. **Run ingestion (first time only):**
   ```bash
   python run.py ingest-comprehensive --clear
   ```

4. **Start the API:**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - Knowledge base statistics
- `POST /ask` - Ask questions about ML documentation
- `GET /docs` - Interactive API documentation

### Example Usage

```bash
curl -X POST "https://your-app.railway.app/ask" \
  -H "Content-Type: application/json" \
  -d '{"q": "How to use PyTorch DataLoader?"}'
```

### Troubleshooting

1. **Health check fails:** Ensure all environment variables are set
2. **No data:** Run ingestion pipeline first
3. **Slow responses:** Consider upgrading Railway plan for more resources

### Cost

- Railway free tier: $5/month credits
- Sufficient for small to medium usage
- Auto-scales based on demand
