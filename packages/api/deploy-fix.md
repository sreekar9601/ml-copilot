# SQLite Database Fix - Deployment Instructions

## Problem Fixed
The production deployment was missing the `bm25.db` SQLite database file, causing the warning:
```
WARNING: SQLite connection not available for keyword search
```

## Solution Applied
Updated both Dockerfiles to include the SQLite database in the Docker image:

### Changes Made:
1. **Dockerfile.api** - Added `COPY data/bm25.db /app/data/bm25.db`
2. **Dockerfile** - Added `COPY ./data/bm25.db /app/data/bm25.db`

## Deployment Steps

### Option 1: Automatic Railway Deployment
If your Railway project is connected to your Git repository:

1. **Commit the changes:**
   ```bash
   git add packages/api/Dockerfile packages/api/Dockerfile.api
   git commit -m "fix: include SQLite database in Docker image for hybrid search"
   git push origin main
   ```

2. **Railway will automatically rebuild and deploy** with the SQLite database included.

### Option 2: Manual Railway Deployment
If you need to deploy manually:

1. **Build and push to Railway:**
   ```bash
   cd packages/api
   railway up
   ```

### Option 3: Local Testing
Test the fix locally first:

1. **Build the Docker image:**
   ```bash
   cd packages/api
   docker build -f Dockerfile.api -t ml-copilot-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key ml-copilot-api
   ```

3. **Test the API:**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"q": "test query", "top_k": 5}'
   ```

## Verification

After deployment, check the logs for:
- ✅ No more "SQLite connection not available" warnings
- ✅ Both vector and keyword search working
- ✅ Hybrid search results with RRF fusion

## Expected Results

The system should now have:
- **Vector Search**: Qdrant Cloud (semantic similarity)
- **Keyword Search**: SQLite FTS5 (BM25 ranking) ✅ **NOW WORKING**
- **Hybrid Search**: RRF fusion of both results ✅ **NOW WORKING**

This will provide more accurate and comprehensive search results for user queries.
