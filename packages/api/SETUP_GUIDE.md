# Setup Guide - Agent System v3.0

> Quick start guide to get the agentic system running

## Prerequisites

- Python 3.11+
- Google Gemini API key
- (Optional) Tavily API key for web search
- (Optional) E2B API key for code execution
- (Optional) LangSmith API key for observability

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd packages/api

# Install all dependencies
pip install -r requirements-api.txt

# Or if you have a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-api.txt
```

### 2. Configure Environment

Create `.env` file from example:

```bash
cp env.example .env
```

Edit `.env` with your keys:

```bash
# Required - Get from Google AI Studio
GOOGLE_API_KEY=your_gemini_api_key_here

# Required - If using Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_key_here

# Optional - For web search
TAVILY_API_KEY=your_tavily_key_here

# Optional - For code execution
E2B_API_KEY=your_e2b_key_here

# Optional - For observability
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
```

### 3. Initialize Database

The agent memory database will be created automatically on first use. To initialize manually:

```bash
python -c "from agent.memory.chat_history import EnhancedChatHistory; h = EnhancedChatHistory('init'); print('✅ Database initialized')"
```

### 4. Test Installation

Run the test script:

```bash
python test_orchestrator.py
```

Expected output:
```
🧪 Testing Agent Orchestrator v3.0
Model: gemini-2.0-flash-exp
--------------------------------------------------

📝 Test Query: How do I create a PyTorch DataLoader?
--------------------------------------------------

🤖 Invoking orchestrator...

✅ Response:
--------------------------------------------------
[Agent's detailed response with code examples and documentation links]
--------------------------------------------------

📊 Metadata:
  Iterations: 2
  Total Cost: $0.0034
  Tools Used: ['hybrid_doc_search']
  Frameworks Detected: ['pytorch']
```

### 5. Verify Components

Test each component individually:

#### Test Configuration
```bash
python -c "from agent.config import config; print(f'Model: {config.orchestrator_model}')"
```

#### Test Memory
```bash
python -c "from agent.memory import EnhancedChatHistory; m = EnhancedChatHistory('test'); print('✅ Memory working')"
```

#### Test Tools
```bash
# Test documentation search
python -c "from agent.tools import hybrid_doc_search; result = hybrid_doc_search('PyTorch'); print(f'Found {result[\"num_results\"]} results')"

# Test syntax validator (no API key needed)
python -c "from agent.tools import validate_code_syntax; result = validate_code_syntax('print(hello)'); print(result)"
```

## Getting API Keys

### Google Gemini API Key (Required)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with Google account
3. Click "Get API Key"
4. Create new API key
5. Copy to `.env` as `GOOGLE_API_KEY`

### Qdrant API Key (Required for production)

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create account
3. Create a cluster
4. Copy URL and API key to `.env`

**Alternative**: Use local ChromaDB (already included)
- No API key needed
- Data stored in `./data/chroma/`

### Tavily API Key (Optional - for web search)

1. Go to [Tavily](https://tavily.com/)
2. Sign up for account
3. Get API key from dashboard
4. Copy to `.env` as `TAVILY_API_KEY`

Without this key:
- Web search tool will return graceful error message
- Agent will still work with documentation search

### E2B API Key (Optional - for code execution)

1. Go to [E2B](https://e2b.dev/)
2. Sign up for account
3. Get API key from dashboard
4. Copy to `.env` as `E2B_API_KEY`

Without this key:
- Code execution tool will return graceful error message
- Syntax validation will still work

### LangSmith API Key (Optional - for observability)

1. Go to [LangSmith](https://smith.langchain.com/)
2. Sign up for account
3. Create new project
4. Get API key
5. Copy to `.env` as `LANGSMITH_API_KEY`

Without this key:
- No tracing/monitoring
- Agent will still work normally

## Minimal Setup (Free Tier)

You can run the agent with just Google Gemini:

```bash
# .env minimal configuration
GOOGLE_API_KEY=your_gemini_key_here
AGENT_ENABLE_WEB_SEARCH=false
AGENT_ENABLE_CODE_EXECUTION=false

# Use local ChromaDB instead of Qdrant
# (if you have existing data)
```

This provides:
- ✅ Documentation search
- ✅ Conversation memory
- ✅ Cost tracking
- ✅ Syntax validation
- ❌ Web search (disabled)
- ❌ Code execution (disabled)

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'langgraph'`

**Solution**:
```bash
pip install -r requirements-api.txt
```

### API Key Errors

**Problem**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**: Make sure `GOOGLE_API_KEY` is set in `.env`:
```bash
echo $GOOGLE_API_KEY  # Should print your key
# Or on Windows:
echo %GOOGLE_API_KEY%
```

### Database Errors

**Problem**: `sqlite3.OperationalError: no such table: chat_history`

**Solution**: Database will auto-create on first use. Force creation:
```bash
python -c "from agent.memory.chat_history import EnhancedChatHistory; EnhancedChatHistory('init')"
```

### ChromaDB Errors

**Problem**: `Collection 'ml_docs' not found`

**Solution**: You need to run the ingestion pipeline first. For testing without data:
```bash
# Disable doc search temporarily
AGENT_ENABLE_DOC_SEARCH=false python test_orchestrator.py
```

Or set up Qdrant with existing data.

## Development Setup

For development with hot-reload:

```bash
# Install dev dependencies
pip install pytest pytest-asyncio black isort mypy

# Run tests
pytest

# Format code
black agent/
isort agent/

# Type checking
mypy agent/
```

## Docker Setup (Optional)

Coming in Phase 2 - FastAPI endpoints with Docker.

## Next Steps

After successful setup:

1. **Try the test script**: `python test_orchestrator.py`
2. **Read the documentation**: `agent/README.md`
3. **Explore examples**: See usage patterns in `agent/README.md`
4. **Check Phase 1 summary**: `PHASE1_IMPLEMENTATION_SUMMARY.md`

## Need Help?

- Check existing code examples in `agent/` directory
- Review error messages (they include helpful suggestions)
- Check the main `AGENT.md` for architecture details

## Production Checklist

Before deploying to production:

- [ ] Set strong API keys
- [ ] Enable LangSmith tracing
- [ ] Set appropriate budget limits
- [ ] Use Qdrant Cloud (not local ChromaDB)
- [ ] Set up monitoring/alerting
- [ ] Test with production data
- [ ] Review cost limits
- [ ] Set up backup for SQLite database

---

**Ready to build! 🚀**

