

# ML Documentation Copilot v3.0 - Deployment Guide

> Complete guide to deploying the agentic system to production

## Overview

This guide covers deploying both the backend API (FastAPI + Agent System) and frontend (Next.js) to production.

## Architecture

```
┌─────────────────┐
│   Frontend      │  Next.js + React
│   (Vercel/      │  • Chat UI
│    Railway)     │  • Streaming SSE
└────────┬────────┘
         │ HTTPS
         ↓
┌─────────────────┐
│   Backend API   │  FastAPI + LangGraph
│   (Railway/     │  • Agent Orchestrator
│    Fly.io)      │  • Tools (RAG, Web, Code)
│                 │  • Memory (SQLite)
│                 │  • Checkpoints (SQLite)
└────────┬────────┘
         │
         ├─→ Qdrant Cloud (Vector DB)
         ├─→ Google Gemini API (LLM)
         ├─→ Tavily API (Web Search)
         └─→ E2B API (Code Execution)
```

---

## Prerequisites

### Required Services

1. **Google Gemini API** (Required)
   - Get API key: https://makersuite.google.com/app/apikey
   - Free tier: 60 requests/minute

2. **Qdrant Cloud** (Required)
   - Create cluster: https://cloud.qdrant.io/
   - Free tier: 1GB cluster

3. **Tavily API** (Optional - for web search)
   - Get API key: https://tavily.com/
   - Free tier: 1000 searches/month

4. **E2B API** (Optional - for code execution)
   - Get API key: https://e2b.dev/
   - Free tier: 100 sandboxes/month

5. **LangSmith** (Optional - for observability)
   - Get API key: https://smith.langchain.com/
   - Free tier: 5000 traces/month

---

## Backend Deployment

### Option 1: Railway (Recommended)

#### 1. Prepare Backend

```bash
cd packages/api

# Create Procfile
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Ensure requirements.txt is up to date
cat requirements-api.txt > requirements.txt
```

#### 2. Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set GOOGLE_API_KEY=<your-key>
railway variables set QDRANT_URL=<your-qdrant-url>
railway variables set QDRANT_API_KEY=<your-qdrant-key>
railway variables set QDRANT_COLLECTION_NAME=ml_docs

# Optional variables
railway variables set TAVILY_API_KEY=<your-tavily-key>
railway variables set E2B_API_KEY=<your-e2b-key>
railway variables set LANGSMITH_API_KEY=<your-langsmith-key>

# Agent configuration
railway variables set AGENT_ORCHESTRATOR_MODEL=gemini-2.0-flash-exp
railway variables set AGENT_MAX_ITERATIONS=10
railway variables set AGENT_MAX_COST_PER_SESSION=0.50
railway variables set AGENT_ENABLE_CODE_EXECUTION=true
railway variables set AGENT_ENABLE_WEB_SEARCH=true

# CORS settings
railway variables set ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Deploy
railway up
```

#### 3. Verify Deployment

```bash
# Get your Railway URL
railway domain

# Test health endpoint
curl https://your-app.up.railway.app/agent/health

# Test agent endpoint
curl -X POST https://your-app.up.railway.app/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PyTorch?"}'
```

### Option 2: Fly.io

#### 1. Create fly.toml

```toml
app = "ml-copilot-api"
primary_region = "iad"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8000"

[[services]]
  http_checks = []
  internal_port = 8000
  processes = ["app"]
  protocol = "tcp"

  [[services.ports]]
    force_https = true
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [[services.tcp_checks]]
    grace_period = "10s"
    interval = "15s"
    restart_limit = 0
    timeout = "2s"

[[mounts]]
  source = "ml_copilot_data"
  destination = "/app/data"
```

#### 2. Deploy

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch app
fly launch

# Set secrets
fly secrets set GOOGLE_API_KEY=<your-key>
fly secrets set QDRANT_URL=<your-qdrant-url>
fly secrets set QDRANT_API_KEY=<your-qdrant-key>
# ... (other secrets)

# Deploy
fly deploy
```

### Option 3: Docker + Any Platform

#### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Build and Run

```bash
# Build
docker build -t ml-copilot-api .

# Run locally
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=<your-key> \
  -e QDRANT_URL=<your-url> \
  -e QDRANT_API_KEY=<your-key> \
  -v $(pwd)/data:/app/data \
  ml-copilot-api

# Push to registry
docker tag ml-copilot-api:latest <registry>/ml-copilot-api:latest
docker push <registry>/ml-copilot-api:latest
```

---

## Frontend Deployment

### Option 1: Vercel (Recommended)

#### 1. Prepare Frontend

```bash
cd packages/web

# Update .env.production
echo "NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app" > .env.production
```

#### 2. Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod

# Or use Vercel GitHub integration (easier)
# 1. Push to GitHub
# 2. Import project in Vercel dashboard
# 3. Set environment variable: NEXT_PUBLIC_API_URL
# 4. Deploy
```

### Option 2: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Build
npm run build

# Deploy
netlify deploy --prod --dir=.next
```

### Option 3: Railway

```bash
cd packages/web

# Create railway.json
cat > railway.json << EOF
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "npm run start",
    "restartPolicyType": "on-failure",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# Deploy
railway init
railway variables set NEXT_PUBLIC_API_URL=<your-backend-url>
railway up
```

---

## Database Setup

### SQLite Persistence

For production, you need to persist the SQLite databases:

#### Railway

```bash
# Mount volume for data persistence
railway volume create data 1

# Update Procfile
echo "web: mkdir -p data && uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile
```

#### Fly.io

Already configured in fly.toml with `mounts`.

#### Docker

```bash
# Use Docker volumes
docker volume create ml-copilot-data

docker run -p 8000:8000 \
  -v ml-copilot-data:/app/data \
  ml-copilot-api
```

### Alternative: Redis (for high scale)

If you need distributed state:

```bash
# Add Redis
railway add redis  # or use managed Redis

# Update .env
AGENT_MEMORY_BACKEND=redis
REDIS_URL=redis://user:pass@host:port
```

---

## Configuration

### Environment Variables Reference

#### Backend (Required)

```bash
# Google Gemini
GOOGLE_API_KEY=<required>

# Qdrant
QDRANT_URL=<required>
QDRANT_API_KEY=<required>
QDRANT_COLLECTION_NAME=ml_docs

# Agent Configuration
AGENT_ORCHESTRATOR_MODEL=gemini-2.0-flash-exp
AGENT_MAX_ITERATIONS=10
AGENT_MAX_COST_PER_SESSION=0.50

# Security
ALLOWED_ORIGINS=https://yourdomain.com
```

#### Backend (Optional)

```bash
# Tools
TAVILY_API_KEY=<optional>
E2B_API_KEY=<optional>
AGENT_ENABLE_WEB_SEARCH=true
AGENT_ENABLE_CODE_EXECUTION=true

# Observability
LANGSMITH_API_KEY=<optional>
LANGSMITH_TRACING=true
```

#### Frontend

```bash
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

---

## Monitoring

### Health Checks

Set up health check endpoints:

```bash
# Railway
railway healthcheck --path /agent/health

# Fly.io (in fly.toml)
[[services.http_checks]]
  interval = "10s"
  grace_period = "5s"
  method = "get"
  path = "/agent/health"
  protocol = "http"
```

### LangSmith Tracing

Enable tracing for production debugging:

```bash
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=ml-copilot-production
LANGSMITH_TRACING=true
```

View traces at: https://smith.langchain.com/

### Application Monitoring

Consider adding:
- **Sentry** for error tracking
- **LogTail** for log aggregation
- **Uptime Robot** for uptime monitoring

---

## Performance Optimization

### 1. CDN for Frontend

Use Vercel/Netlify CDN by default.

### 2. API Response Caching

```python
# Add caching for frequent queries
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

### 3. Database Optimization

```bash
# Regular SQLite vacuum
sqlite3 data/agent_memory.db "VACUUM;"

# Or add to cron
echo "0 2 * * * sqlite3 /app/data/agent_memory.db 'VACUUM;'" | crontab -
```

---

## Security

### 1. CORS Configuration

```python
# Only allow your frontend domain
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 2. Rate Limiting

```python
# Already implemented in main.py
# Adjust limits in .env
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

### 3. API Keys

```bash
# Never commit API keys
# Use environment variables
# Rotate keys regularly
```

### 4. HTTPS

- Railway/Fly.io provide HTTPS automatically
- Vercel provides HTTPS by default
- Custom domains: Use Let's Encrypt

---

## Backup & Recovery

### Database Backups

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
railway run sqlite3 data/agent_memory.db ".backup /tmp/backup_$DATE.db"
railway run cp /tmp/backup_$DATE.db /backups/

# Automate with cron
```

### Restore

```bash
# Download backup
railway run cp /backups/backup_YYYYMMDD_HHMMSS.db data/agent_memory.db

# Restart service
railway restart
```

---

## Troubleshooting

### Common Issues

#### 1. CORS Errors

```bash
# Check ALLOWED_ORIGINS
railway logs | grep CORS

# Update
railway variables set ALLOWED_ORIGINS=https://yourfrontend.vercel.app
```

#### 2. API Connection Timeout

```bash
# Increase timeout in frontend
// fetch options
{ timeout: 120000 }  // 2 minutes
```

#### 3. High Costs

```bash
# Reduce max cost per session
railway variables set AGENT_MAX_COST_PER_SESSION=0.25

# Disable expensive features
railway variables set AGENT_ENABLE_CODE_EXECUTION=false
```

#### 4. Database Lock Errors

```bash
# SQLite WAL mode (already enabled)
# Or switch to Redis
railway variables set AGENT_MEMORY_BACKEND=redis
```

---

## Scaling

### Horizontal Scaling

```bash
# Railway
railway scale --replicas 3

# Fly.io
fly scale count 3
```

### Vertical Scaling

```bash
# Railway
railway scale --size M  # or L, XL

# Fly.io
fly scale vm shared-cpu-2x
```

### Requirements:
- Use Redis for memory backend
- Use shared checkpoint storage (S3/Cloudflare R2)

---

## Cost Optimization

### Estimated Costs (per month)

| Service | Free Tier | Paid |
|---------|-----------|------|
| **Backend Hosting** (Railway) | $5 credit | ~$10-20 |
| **Frontend Hosting** (Vercel) | Free | $20+ |
| **Qdrant Cloud** | 1GB free | ~$25/month |
| **Google Gemini** | 60 req/min free | Pay per use |
| **Tavily** (optional) | 1000 free | ~$29/month |
| **E2B** (optional) | 100 free | ~$20/month |
| **LangSmith** (optional) | 5000 traces free | ~$39/month |

**Total**: $0-15/month (free tiers) or $100+/month (full features)

### Tips:
1. Start with free tiers
2. Monitor usage with LangSmith
3. Set budget limits in `.env`
4. Disable expensive features if not needed

---

## Success Checklist

- [ ] Backend deployed and accessible
- [ ] Frontend deployed and accessible
- [ ] Health endpoint returns 200
- [ ] Test query works via API
- [ ] Streaming works
- [ ] Conversation history persists
- [ ] CORS configured correctly
- [ ] HTTPS enabled
- [ ] Monitoring set up
- [ ] Backups configured
- [ ] Cost limits set

---

## Support

- **Documentation**: See `PHASE2_WEEKS5-6_SUMMARY.md` and `API_USAGE_GUIDE.md`
- **Issues**: Check error logs with `railway logs` or `fly logs`
- **Performance**: Monitor with LangSmith traces

---

**Ready for production! 🚀**

