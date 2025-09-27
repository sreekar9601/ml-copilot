# Railway + Qdrant Cloud Deployment Guide

This guide walks you through deploying the ML Documentation Copilot to **Railway** with **Qdrant Cloud** as the vector database.

## 🚀 **Architecture Overview**

- **API Service**: Railway (lightweight FastAPI app)
- **Vector Database**: Qdrant Cloud (managed)
- **Keyword Search**: SQLite FTS5 (deployed with app)
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google Cloud Vertex AI (text-embedding-004)

## 📋 **Prerequisites**

1. **GitHub Account** - For code repository
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Qdrant Cloud Account** - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
4. **Google Cloud Account** - For Gemini API ([ai.google.dev](https://ai.google.dev))

## 🔧 **Step 1: Set Up Qdrant Cloud**

### Create Qdrant Cluster
1. Go to [Qdrant Cloud Console](https://cloud.qdrant.io)
2. Click **"Create Cluster"**
3. Choose:
   - **Plan**: Free tier (1GB storage)
   - **Region**: Closest to your users
   - **Cluster Name**: `ml-docs-copilot`
4. Wait for cluster provisioning (~2-3 minutes)

### Get Connection Details
1. Once cluster is ready, click on it
2. Copy the **Cluster URL** (format: `https://xxx-xxx-xxx.qdrant.tech`)
3. Click **"API Keys"** → **"Create API Key"**
4. Copy the **API Key** (starts with `qnt_`)

## 🔧 **Step 2: Set Up Google Cloud**

### Get Gemini API Key
1. Go to [Google AI Studio](https://ai.google.dev)
2. Click **"Get API Key"**
3. Create a new project or select existing
4. Copy your **API Key** (starts with `AIza`)

## 🔧 **Step 3: Prepare GitHub Repository**

### Push to GitHub
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial ML Documentation Copilot setup"

# Add your GitHub repository
git remote add origin https://github.com/yourusername/ml-docs-copilot.git
git branch -M main
git push -u origin main
```

### Create .env.example (already updated)
The `.env.example` file contains all necessary environment variables:

```env
# Google Gemini API Key (required)
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant Cloud Configuration (for production)
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=ml_docs

# Local Data Storage (fallback/development)
DATA_DIR=./data

# SQLite database file name
SQLITE_DB=bm25.db
```

## 🚀 **Step 4: Deploy to Railway**

### Connect GitHub to Railway
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub account and select your repository

### Configure Environment Variables
In Railway project settings → **Environment** tab, add:

```env
GOOGLE_API_KEY=AIza...your_actual_key
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=qnt_...your_actual_key
QDRANT_COLLECTION_NAME=ml_docs
DATA_DIR=/app/data
SQLITE_DB=bm25.db
PORT=8000
```

### Railway Configuration
Railway will automatically detect the `Dockerfile.api` and build configuration. Key files:

- **Dockerfile.api**: Optimized for lightweight API deployment
- **railway.json**: Specifies Dockerfile and build settings
- **nixpacks.toml**: Alternative Nixpacks configuration (if needed)

### Deploy Settings
1. **Build Command**: Automatic (Docker)
2. **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. **Root Directory**: `/` (project root)

## 📊 **Step 5: Data Ingestion**

### Option A: Local Ingestion + Upload to Qdrant
```bash
# 1. Set up local environment with Qdrant credentials
cp env.example .env
# Edit .env with your actual credentials

# 2. Run ingestion locally
python ingest/ingest_focused_web.py

# 3. Data will be uploaded to Qdrant Cloud automatically
```

### Option B: Remote Ingestion Service
Deploy the ingestion service separately:
```bash
# Deploy ingestion-service/ as a separate Railway service
# Run ingestion jobs on-demand
```

## 🧪 **Step 6: Test Deployment**

### Health Check
```bash
curl https://your-railway-app.railway.app/health
```

### Database Stats
```bash
curl https://your-railway-app.railway.app/stats
```

### Test Query
```bash
curl -X POST https://your-railway-app.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I implement transfer learning in TensorFlow?"}'
```

## 🔍 **Step 7: Monitoring & Maintenance**

### Railway Monitoring
- **Logs**: Railway dashboard → Deployments → View logs
- **Metrics**: CPU, memory, response times
- **Scaling**: Auto-scaling based on traffic

### Qdrant Cloud Monitoring
- **Storage Usage**: Monitor free tier limits (1GB)
- **Query Performance**: Track search latency
- **API Usage**: Monitor rate limits

### Cost Management
- **Railway**: Pay-per-use, ~$5-10/month for typical usage
- **Qdrant Cloud**: Free tier (1GB), then $10+/month
- **Google Gemini**: Pay-per-token, very affordable for documentation queries

## 🛠 **Troubleshooting**

### Common Issues

#### Build Timeouts
- Ensure `.dockerignore` excludes large directories
- Use optimized `Dockerfile.api` (not the full `Dockerfile`)

#### Qdrant Connection Issues
- Verify cluster URL and API key
- Check network connectivity from Railway
- Ensure collection name matches

#### Google API Errors
- Verify API key is valid and has Gemini access
- Check quota limits
- Ensure billing is enabled (if required)

#### Empty Search Results
- Check if data was properly ingested to Qdrant
- Verify collection exists and has vectors
- Test with simple queries first

### Debug Commands
```bash
# Check Railway logs
railway logs

# Test local container
docker run --rm -p 8000:8000 \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  -e QDRANT_URL=$QDRANT_URL \
  -e QDRANT_API_KEY=$QDRANT_API_KEY \
  your-image-name

# Test Qdrant connection
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='$QDRANT_URL', api_key='$QDRANT_API_KEY')
print('Collections:', client.get_collections())
"
```

## 🎯 **Production Checklist**

- [ ] Qdrant Cloud cluster provisioned
- [ ] Google Gemini API key obtained
- [ ] Environment variables configured in Railway
- [ ] GitHub repository connected to Railway
- [ ] Successful deployment to Railway
- [ ] Data ingested to Qdrant Cloud
- [ ] Health checks passing
- [ ] Test queries returning results
- [ ] Monitoring set up
- [ ] Domain configured (optional)

## 📈 **Scaling Considerations**

### Performance
- **Vector Search**: Qdrant Cloud handles scaling automatically
- **API Scaling**: Railway auto-scales based on traffic
- **Caching**: Consider Redis for frequent queries

### Storage
- **Free Tier**: 330k vectors (~1GB)
- **Upgrade**: $10/month for more storage
- **Optimization**: Reduce vector dimensions if needed

### Security
- **API Keys**: Store securely in Railway environment variables
- **Rate Limiting**: Implement if needed for public deployment
- **CORS**: Configure for web frontend integration

---

🎉 **Your ML Documentation Copilot is now running on Railway with Qdrant Cloud!**
