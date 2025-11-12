# Agent API Usage Guide

> Complete guide to using the ML Documentation Copilot v3.0 API

## Quick Start

### Start the Server

```bash
cd packages/api
uvicorn api.main:app --reload --port 8000
```

Server will be available at: `http://localhost:8000`

API Documentation (Swagger): `http://localhost:8000/docs`

---

## Endpoints Overview

| Endpoint | Method | Description | Streaming |
|----------|--------|-------------|-----------|
| `/agent/invoke` | POST | Ask a question | No |
| `/agent/invoke/stream` | POST | Ask with real-time updates | Yes (SSE) |
| `/agent/conversations/{id}` | GET | Get conversation history | No |
| `/agent/conversations/{id}` | DELETE | Clear conversation | No |
| `/agent/health` | GET | Health check | No |

---

## 1. Non-Streaming Request

### Simple Query

```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a PyTorch DataLoader?"
  }'
```

**Response:**
```json
{
  "response": "To create a PyTorch DataLoader, you need to...",
  "conversation_id": "abc-123-def",
  "metadata": {
    "iterations": 2,
    "total_cost": 0.0034,
    "tools_used": ["hybrid_doc_search"],
    "frameworks_detected": ["pytorch"]
  }
}
```

### Multi-Turn Conversation

```bash
# First message
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is PyTorch?",
    "conversation_id": "my-session"
  }'

# Follow-up (remembers context)
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me an example",
    "conversation_id": "my-session"
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/agent/invoke",
    json={
        "query": "How to use MLflow model registry?",
        "conversation_id": "optional-session-id"
    }
)

data = response.json()
print(data["response"])
print(f"Cost: ${data['metadata']['total_cost']:.4f}")
print(f"Tools used: {', '.join(data['metadata']['tools_used'])}")
```

### JavaScript Example

```javascript
async function askAgent(query) {
  const response = await fetch('http://localhost:8000/agent/invoke', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  
  const data = await response.json();
  console.log(data.response);
  return data;
}

askAgent("What is Ray Serve?");
```

---

## 2. Streaming Request (SSE)

### JavaScript EventSource

```javascript
const query = "How do I create a PyTorch DataLoader?";

const eventSource = new EventSource(
  `http://localhost:8000/agent/invoke/stream`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  }
);

// Listen for different event types
eventSource.addEventListener('thought', (e) => {
  const data = JSON.parse(e.data);
  console.log('💭 Thinking:', data.content);
});

eventSource.addEventListener('tool_call', (e) => {
  const data = JSON.parse(e.data);
  console.log('🔧 Tool:', data.display);
});

eventSource.addEventListener('token', (e) => {
  const data = JSON.parse(e.data);
  process.stdout.write(data.content);  // Real-time typing
});

eventSource.addEventListener('done', (e) => {
  const data = JSON.parse(e.data);
  console.log('\n✅ Done!', data.metadata);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  console.error('❌ Error:', data.error);
  eventSource.close();
});
```

### Python httpx Streaming

```python
import httpx
import json

async def stream_agent(query: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/agent/invoke/stream",
            json={"query": query},
            timeout=120.0
        ) as response:
            current_event = None
            
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    
                    if current_event == "token":
                        print(data["content"], end="", flush=True)
                    elif current_event == "done":
                        print(f"\n\nCost: ${data['metadata']['cost']:.4f}")

# Usage
import asyncio
asyncio.run(stream_agent("What is PyTorch?"))
```

### Fetch API with Manual Parsing

```javascript
async function streamAgent(query) {
  const response = await fetch('http://localhost:8000/agent/invoke/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // Keep incomplete line

    for (const line of lines) {
      if (line.startsWith('event:')) {
        const eventType = line.slice(6).trim();
        // Handle event type
      } else if (line.startsWith('data:')) {
        const data = JSON.parse(line.slice(5).trim());
        // Handle data
      }
    }
  }
}
```

---

## 3. Conversation Management

### Get Conversation History

```bash
curl http://localhost:8000/agent/conversations/my-session
```

**Response:**
```json
{
  "conversation_id": "my-session",
  "messages": [
    {
      "type": "human",
      "content": "What is PyTorch?"
    },
    {
      "type": "ai",
      "content": "PyTorch is a deep learning framework..."
    }
  ],
  "stats": {
    "message_count": 4,
    "total_tokens": 1250,
    "total_cost": 0.0089,
    "started_at": "2025-01-15 10:30:00",
    "last_activity": "2025-01-15 10:35:22"
  }
}
```

### Clear Conversation

```bash
curl -X DELETE http://localhost:8000/agent/conversations/my-session
```

**Response:**
```json
{
  "status": "cleared",
  "conversation_id": "my-session",
  "message": "Conversation history has been deleted"
}
```

---

## 4. Health Check

```bash
curl http://localhost:8000/agent/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "agent_model": "gemini-2.0-flash-exp",
  "features": {
    "code_execution": true,
    "web_search": true,
    "self_reflection": false
  },
  "limits": {
    "max_iterations": 10,
    "max_cost_per_session": 0.50
  }
}
```

---

## Event Types Reference

### Streaming Events

| Event | Description | Example Data |
|-------|-------------|--------------|
| `thought` | Agent is thinking | `{"content": "🧠 Analyzing..."}` |
| `tool_call` | Tool is being called | `{"tool": "hybrid_doc_search", "display": "🔍 Searching", "status": "running"}` |
| `tool_result` | Tool completed | `{"tool": "hybrid_doc_search", "preview": "Found 5 results", "status": "complete"}` |
| `token` | Response token | `{"content": "Hello "}` |
| `done` | Stream completed | `{"conversation_id": "abc", "metadata": {...}}` |
| `error` | Error occurred | `{"error": "Budget exceeded", "type": "BudgetExceededError"}` |

---

## Error Handling

### Common Errors

#### Budget Exceeded
```json
{
  "detail": "Agent execution failed: Session cost $0.51 exceeds limit $0.50"
}
```

**Solution:** Start new conversation or increase `AGENT_MAX_COST_PER_SESSION`

#### Max Iterations Reached
```json
{
  "detail": "Agent execution failed: Maximum iterations reached"
}
```

**Solution:** Simplify query or increase `AGENT_MAX_ITERATIONS`

#### Connection Error
```json
{
  "detail": "Failed to connect to Qdrant"
}
```

**Solution:** Check `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

---

## Best Practices

### 1. Use Conversation IDs

```javascript
// Good: Maintains context
const sessionId = generateUUID();
await askAgent("What is PyTorch?", sessionId);
await askAgent("Show me an example", sessionId);  // Remembers context

// Bad: No context
await askAgent("What is PyTorch?");
await askAgent("Show me an example");  // No memory of previous question
```

### 2. Prefer Streaming for UX

```javascript
// Better UX: User sees progress immediately
streamAgent(query);  // Shows thinking, tools, tokens in real-time

// Worse UX: User waits 5-10 seconds
askAgent(query);  // Shows nothing until complete
```

### 3. Handle Errors Gracefully

```javascript
try {
  const response = await askAgent(query);
  displayResponse(response.response);
} catch (error) {
  if (error.message.includes("Budget exceeded")) {
    showMessage("Session budget reached. Starting new conversation...");
    const response = await askAgent(query, generateNewSessionId());
  } else {
    showError(error.message);
  }
}
```

### 4. Monitor Costs

```javascript
const response = await askAgent(query);
const cost = response.metadata.total_cost;

// Track cumulative cost
totalCost += cost;
if (totalCost > WARNING_THRESHOLD) {
  showWarning(`Session cost: $${totalCost.toFixed(4)}`);
}
```

---

## Testing

### Run Test Suite

```bash
python test_streaming.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/agent/health

# Simple query
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PyTorch?"}'

# Check logs
tail -f logs/agent.log
```

---

## Configuration

### Environment Variables

```bash
# .env file
AGENT_ORCHESTRATOR_MODEL=gemini-2.0-flash-exp
AGENT_MAX_ITERATIONS=10
AGENT_MAX_COST_PER_SESSION=0.50
AGENT_ENABLE_WEB_SEARCH=true
AGENT_ENABLE_CODE_EXECUTION=true
```

### Adjust Limits

```bash
# Increase budget
AGENT_MAX_COST_PER_SESSION=1.00

# More iterations for complex queries
AGENT_MAX_ITERATIONS=20

# Disable expensive features
AGENT_ENABLE_CODE_EXECUTION=false
```

---

## Production Deployment

### CORS Configuration

Update `ALLOWED_ORIGINS` in `.env`:

```bash
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Rate Limiting

The API includes built-in rate limiting (60 requests/minute by default).

### Monitoring

Health check endpoint for monitoring:

```bash
curl http://your-domain.com/agent/health
```

Monitor:
- `status`: Should be "healthy"
- Response time: Should be < 1 second
- Features enabled

### Load Balancing

The agent system uses SQLite for checkpoints and memory. For multi-instance deployments:

1. Use Redis backend: `AGENT_MEMORY_BACKEND=redis`
2. Configure shared checkpoint storage
3. Use sticky sessions (conversation_id-based routing)

---

## Examples Gallery

### Example 1: Simple Q&A

**Query:** "What is PyTorch?"  
**Tools Used:** hybrid_doc_search  
**Response Time:** ~2s  
**Cost:** $0.002

### Example 2: Code Example with Validation

**Query:** "Show me how to create a DataLoader and test it"  
**Tools Used:** hybrid_doc_search, execute_python_code  
**Response Time:** ~8s  
**Cost:** $0.015

### Example 3: Recent Information

**Query:** "What are the breaking changes in PyTorch 2.5?"  
**Tools Used:** web_search, hybrid_doc_search  
**Response Time:** ~6s  
**Cost:** $0.008

### Example 4: Multi-Turn Conversation

**Query 1:** "Explain PyTorch tensors"  
**Query 2:** "Show me an example"  
**Query 3:** "How do I move it to GPU?"  
**Total Cost:** $0.012

---

## Troubleshooting

### No Response from Server

```bash
# Check if server is running
curl http://localhost:8000/agent/health

# Check logs
tail -f api.log
```

### Streaming Not Working

```bash
# Test with curl
curl -N -X POST http://localhost:8000/agent/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### High Costs

```bash
# Check conversation stats
curl http://localhost:8000/agent/conversations/your-session-id

# Clear expensive conversations
curl -X DELETE http://localhost:8000/agent/conversations/your-session-id
```

---

**Need help? Check the comprehensive documentation in `PHASE2_WEEKS5-6_SUMMARY.md`**

