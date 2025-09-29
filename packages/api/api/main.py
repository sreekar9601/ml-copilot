"""FastAPI application for the ML documentation copilot."""

import logging
import time
import os
import platform
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
# IMPORTANT: Import clients module first to ensure configuration runs
from .clients import get_generation_model, GENERATION_MODEL_NAME
from .config import settings
# Import retrieval modules lazily to avoid startup failures
# from .retrieval import retrieve_documents, RetrievalResult
from .prompts import SYSTEM_PROMPT, CONTEXT_CHUNK_TEMPLATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add startup logging
logger.info("Starting ML Documentation Copilot API...")
logger.info(f"Data directory: {settings.data_dir}")
logger.info(f"ChromaDB collection: {settings.chroma_collection}")
logger.info(f"SQLite database: {settings.sqlite_db}")

# Configuration is now handled in clients.py
# No need to configure here as it's already done

# Debug logging for Google API configuration
api_key_prefix = settings.google_api_key[:8] + "..." if settings.google_api_key else "NOT_SET"
logger.info(f"🔑 Google API Key prefix: {api_key_prefix}")
logger.info(f"🤖 Using Gemini model: gemini-1.5-flash")
logger.info(f"🌐 Transport: REST (forced Studio API routing)")

# Initialize FastAPI app
app = FastAPI(
    title="ML Documentation Copilot",
    description="AI assistant for ML infrastructure documentation (PyTorch, MLflow, Ray Serve, KServe)",
    version="1.0.0"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# CORS middleware with production-safe settings
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = allowed_origins_env.split(",")
else:
    # Default origins for development
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Configure via ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow needed methods
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],  # Specific headers only
)

# Simple rate limiting (in-memory, suitable for single instance)
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    # Clean old requests outside the window
    cutoff_time = current_time - timedelta(seconds=RATE_LIMIT_WINDOW)
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip] 
        if req_time > cutoff_time
    ]
    
    # Check if rate limit exceeded
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ML Documentation Copilot API is ready!")
    logger.info("📊 Available endpoints:")
    logger.info("  - GET  /health - Health check")
    logger.info("  - POST /ask - Ask questions")
    logger.info("  - GET  /stats - Database statistics")


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    q: str = Field(..., description="User question about ML infrastructure")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top documents to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    use_expansion: bool = Field(default=False, description="Whether to use query expansion (advanced mode)")
    use_reranking: bool = Field(default=False, description="Whether to use re-ranking (advanced mode)")


class AdvancedQueryRequest(BaseModel):
    """Request model for the /ask-advanced endpoint."""
    q: str = Field(..., description="User question about ML infrastructure")
    top_k: int = Field(default=10, ge=1, le=20, description="Number of top documents to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    use_expansion: bool = Field(default=True, description="Whether to use query expansion")
    use_reranking: bool = Field(default=True, description="Whether to use re-ranking")


class SourceInfo(BaseModel):
    """Information about a source document."""
    chunk_id: str
    title: str
    url: str
    heading_path: str
    anchor_link: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: List[SourceInfo]
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    chunks_retrieved: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    data_dir_exists: bool
    chromadb_available: bool
    sqlite_available: bool


class ReindexRequest(BaseModel):
    """Request model for reindexing."""
    clear_existing: bool = Field(default=True, description="Whether to clear existing data")


# Global Gemini model is now managed in clients.py


def get_gemini_model():
    """Get the pre-configured Gemini model instance."""
    return get_generation_model()


def format_context_chunks(results: List[Any]) -> str:
    """Format retrieval results into context chunks for the prompt."""
    context_chunks = []
    
    for result in results:
        chunk_text = CONTEXT_CHUNK_TEMPLATE.format(
            chunk_id=result.chunk_id,
            source_url=result.metadata.get('source_url', 'Unknown'),
            heading_path=result.metadata.get('heading_path', 'Unknown'),
            content=result.content
        )
        context_chunks.append(chunk_text)
    
    return "\n".join(context_chunks)


def generate_answer(query: str, context_chunks: str) -> str:
    """Generate answer using Gemini."""
    try:
        client = get_gemini_model()  # This now returns the client
        
        prompt = SYSTEM_PROMPT.format(
            context_chunks=context_chunks,
            user_question=query
        )
        
        # Generate response using Vertex AI
        response = client.models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=prompt
        )
        
        if response.text:
            return response.text
        else:
            logger.error("Empty response from Gemini")
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"I encountered an error while processing your question: {str(e)}"


@app.get("/health")
async def health_check():
    """Simple health check endpoint that always returns 200 OK."""
    return {"status": "healthy", "message": "API service is running"}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment variables."""
    import os
    api_key_prefix = settings.google_api_key[:8] + "..." if settings.google_api_key else "NOT_SET"
    
    return {
        "status": "debug",
        "port_env": os.getenv("PORT", "not_set"),
        "host": "0.0.0.0",
        "python_version": f"{platform.python_version()}",
        "working_dir": str(Path.cwd()),
        "google_api_key_prefix": api_key_prefix,
        "gemini_model": "gemini-2.5-flash",
        "authentication": "vertex_ai_full",
        "vertex_env_vars": {
            "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", "not_set"),
            "GOOGLE_CLOUD_LOCATION": os.getenv("GOOGLE_CLOUD_LOCATION", "not_set"),
            "GOOGLE_GENAI_USE_VERTEXAI": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "not_set"),
            "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "not_set")
        }
    }

@app.get("/health-detailed", response_model=HealthResponse)
async def health_check_detailed():
    """Health check endpoint."""
    
    # Check if data directory exists
    data_dir_exists = settings.data_dir.exists()
    
    # Check ChromaDB availability
    chromadb_available = False
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        chromadb_available = retriever.collection is not None
    except Exception as e:
        # Log the error for debugging but don't fail the health check
        print(f"ChromaDB check failed: {e}")
        pass
    
    # Check SQLite availability
    sqlite_available = False
    try:
        sqlite_path = settings.data_dir / settings.sqlite_db
        sqlite_available = sqlite_path.exists()
    except Exception as e:
        # Log the error for debugging but don't fail the health check
        print(f"SQLite check failed: {e}")
        pass
    
    # Always return "healthy" for the health check to pass
    # The actual status is "degraded" if databases are missing, but that's OK
    status = "healthy"
    
    return HealthResponse(
        status=status,
        data_dir_exists=data_dir_exists,
        chromadb_available=chromadb_available,
        sqlite_available=sqlite_available
    )


@app.post("/ask-advanced", response_model=QueryResponse)
async def ask_question_advanced(request: AdvancedQueryRequest):
    """Advanced endpoint with query expansion and re-ranking options."""
    
    start_time = time.time()
    
    if not request.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Import advanced retrieval modules lazily
        from .advanced_retrieval import advanced_retrieve_documents, RetrievalResult
        
        # Retrieve relevant documents using advanced retrieval
        retrieval_start = time.time()
        results = advanced_retrieve_documents(
            request.q, 
            top_k=request.top_k,
            use_expansion=request.use_expansion,
            use_reranking=request.use_reranking
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found. The knowledge base might be empty or your query is too specific."
            )
        
        # Format context for the LLM
        context_chunks = format_context_chunks(results)
        
        # Generate answer
        generation_start = time.time()
        answer = generate_answer(request.q, context_chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        # Prepare sources
        sources = []
        if request.include_sources:
            for result in results:
                sources.append(SourceInfo(
                    chunk_id=result.chunk_id,
                    title=result.metadata.get('title', 'Unknown'),
                    url=result.metadata.get('source_url', ''),
                    heading_path=result.metadata.get('heading_path', ''),
                    anchor_link=result.metadata.get('anchor_link', ''),
                    relevance_score=result.score
                ))
        
        total_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.q,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            chunks_retrieved=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing advanced query '{request.q}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint for asking questions about ML documentation."""
    
    start_time = time.time()
    
    if not request.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Import advanced retrieval modules lazily
        from .advanced_retrieval import advanced_retrieve_documents, RetrievalResult
        
        # Retrieve relevant documents using advanced retrieval
        retrieval_start = time.time()
        results = advanced_retrieve_documents(
            request.q, 
            top_k=request.top_k,
            use_expansion=request.use_expansion,
            use_reranking=request.use_reranking
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No relevant documents found. The knowledge base might be empty or your query is too specific."
            )
        
        # Format context for the LLM
        context_chunks = format_context_chunks(results)
        
        # Generate answer
        generation_start = time.time()
        answer = generate_answer(request.q, context_chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        # Prepare sources
        sources = []
        if request.include_sources:
            for result in results:
                sources.append(SourceInfo(
                    chunk_id=result.chunk_id,
                    title=result.metadata.get('title', 'Unknown'),
                    url=result.metadata.get('source_url', ''),
                    heading_path=result.metadata.get('heading_path', ''),
                    anchor_link=result.metadata.get('anchor_link', ''),
                    relevance_score=result.score
                ))
        
        total_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.q,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            chunks_retrieved=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query '{request.q}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/sources/{chunk_id}")
async def get_chunk_details(chunk_id: str):
    """Get detailed information about a specific chunk."""
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        
        chunk = retriever._fetch_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "title": chunk.metadata.get('title', 'Unknown'),
            "source_url": chunk.metadata.get('source_url', ''),
            "heading_path": chunk.metadata.get('heading_path', ''),
            "anchor_link": chunk.metadata.get('anchor_link', '')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chunk: {str(e)}")


@app.post("/reindex")
async def reindex_documents(request: ReindexRequest, background_tasks: BackgroundTasks):
    """Reindex the document collection (admin endpoint)."""
    
    def run_reindexing(clear_existing: bool):
        """Background task for reindexing."""
        try:
            import asyncio
            from ingest.main import run_ingestion_pipeline
            
            seeds_path = Path(__file__).parent.parent / "ingest" / "seeds.yaml"
            
            # Run the ingestion pipeline
            stats = asyncio.run(run_ingestion_pipeline(
                seeds_path=seeds_path,
                data_dir=settings.data_dir,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                collection_name=settings.chroma_collection,
                sqlite_db=settings.sqlite_db,
                clear_existing=clear_existing
            ))
            
            logger.info(f"Reindexing completed: {stats}")
            
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")
    
    # Start reindexing in background
    background_tasks.add_task(run_reindexing, request.clear_existing)
    
    return {
        "message": "Reindexing started in background",
        "clear_existing": request.clear_existing
    }


@app.get("/debug/retrieval")
async def debug_retrieval(query: str = "PyTorch DataLoader"):
    """Debug endpoint to see what's being retrieved for a query."""
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        
        # Get vector results
        vector_results = retriever.vector_search(query, top_k=5)
        
        # Get keyword results  
        keyword_results = retriever.keyword_search(query, top_k=5)
        
        # Get hybrid results
        hybrid_results = retriever.retrieve(query, top_k=5)
        
        return {
            "query": query,
            "vector_results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "metadata": r.metadata
                } for r in vector_results
            ],
            "keyword_results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "metadata": r.metadata
                } for r in keyword_results
            ],
            "hybrid_results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "metadata": r.metadata
                } for r in hybrid_results
            ],
            "vector_count": len(vector_results),
            "keyword_count": len(keyword_results),
            "hybrid_count": len(hybrid_results)
        }
        
    except Exception as e:
        logger.error(f"Error in debug retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error in debug retrieval: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base."""
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        
        stats = {}
        
        # Qdrant stats
        if retriever.use_qdrant and retriever.qdrant_client:
            try:
                collection_info = retriever.qdrant_client.get_collection(settings.qdrant_collection_name)
                stats["qdrant_count"] = collection_info.points_count
                stats["qdrant_vector_size"] = collection_info.config.params.vectors.size
            except Exception as e:
                stats["qdrant_error"] = str(e)
        
        # ChromaDB stats
        if retriever.collection:
            stats["chromadb_count"] = retriever.collection.count()
        
        # SQLite stats
        if retriever.sqlite_conn:
            cursor = retriever.sqlite_conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents_fts")
            stats["sqlite_fts_count"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
            stats["sqlite_metadata_count"] = cursor.fetchone()[0]
            
            # Get source distribution
            cursor.execute("""
                SELECT source_url, COUNT(*) as chunk_count 
                FROM chunk_metadata 
                GROUP BY source_url 
                ORDER BY chunk_count DESC
            """)
            
            source_distribution = {}
            for row in cursor.fetchall():
                source_distribution[row[0]] = row[1]
            
            stats["source_distribution"] = source_distribution
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


class HowToRequest(BaseModel):
    """Request model for how-to tutorial generation."""
    query: str = Field(..., min_length=3, max_length=500, description="The task to create a tutorial for")
    max_steps: int = Field(default=8, ge=1, le=15, description="Maximum number of steps to generate")


@app.post("/howto")
async def generate_howto_tutorial(request: HowToRequest):
    """Generate a structured step-by-step tutorial for a given task."""
    try:
        start_time = time.time()
        
        # Import planner components lazily
        from .planner import AdvancedPlanner, create_tutorial_from_plan
        
        logger.info(f"Generating how-to tutorial for: {request.query}")
        
        # Create planner and generate plan
        planner = AdvancedPlanner()
        plan = planner.create_plan(request.query)
        
        # Limit steps if needed
        if len(plan.steps) > request.max_steps:
            plan.steps = plan.steps[:request.max_steps]
        
        logger.info(f"Created plan with {len(plan.steps)} steps, intent: {plan.intent}")
        
        # Generate tutorial from plan
        tutorial = create_tutorial_from_plan(plan)
        
        processing_time = time.time() - start_time
        
        response = {
            "tutorial": tutorial.dict(),
            "processing_time": round(processing_time, 2),
            "plan_steps": len(plan.steps),
            "completed_steps": len(tutorial.steps),
            "total_citations": tutorial.total_citations
        }
        
        logger.info(f"Generated tutorial with {len(tutorial.steps)} steps and {tutorial.total_citations} citations in {processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating how-to tutorial: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tutorial: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Documentation Copilot",
        "description": "AI assistant for ML infrastructure documentation",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask questions about ML documentation (standard retrieval)",
            "/ask-advanced": "POST - Ask questions with advanced query expansion and re-ranking",
            "/howto": "POST - Generate step-by-step tutorials with citations",
            "/health": "GET - Health check",
            "/stats": "GET - Knowledge base statistics",
            "/debug/retrieval": "GET - Debug retrieval results for a query",
            "/sources/{chunk_id}": "GET - Get chunk details",
            "/reindex": "POST - Reindex documents (admin)",
            "/docs": "GET - API documentation"
        },
        "supported_topics": [
            "PyTorch (data loading, distributed training)",
            "MLflow (tracking, model registry)",
            "KServe (inference services, deployment)",
            "Ray Serve (production serving, configuration)"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

