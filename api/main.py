"""FastAPI application for the ML documentation copilot."""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

from .config import settings
from .retrieval import retrieve_documents, RetrievalResult
from .prompts import SYSTEM_PROMPT, CONTEXT_CHUNK_TEMPLATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=settings.google_api_key)

# Initialize FastAPI app
app = FastAPI(
    title="ML Documentation Copilot",
    description="AI assistant for ML infrastructure documentation (PyTorch, MLflow, Ray Serve, KServe)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    q: str = Field(..., description="User question about ML infrastructure")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top documents to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source information")


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


# Global Gemini model
gemini_model = None


def get_gemini_model():
    """Get or create Gemini model instance."""
    global gemini_model
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 2048,
            }
        )
    return gemini_model


def format_context_chunks(results: List[RetrievalResult]) -> str:
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
        model = get_gemini_model()
        
        prompt = SYSTEM_PROMPT.format(
            context_chunks=context_chunks,
            user_question=query
        )
        
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            logger.error("Empty response from Gemini")
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"I encountered an error while processing your question: {str(e)}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    # Check if data directory exists
    data_dir_exists = settings.data_dir.exists()
    
    # Check ChromaDB availability
    chromadb_available = False
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        chromadb_available = retriever.collection is not None
    except Exception:
        pass
    
    # Check SQLite availability
    sqlite_available = False
    try:
        sqlite_path = settings.data_dir / settings.sqlite_db
        sqlite_available = sqlite_path.exists()
    except Exception:
        pass
    
    status = "healthy" if (data_dir_exists and chromadb_available and sqlite_available) else "degraded"
    
    return HealthResponse(
        status=status,
        data_dir_exists=data_dir_exists,
        chromadb_available=chromadb_available,
        sqlite_available=sqlite_available
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint for asking questions about ML documentation."""
    
    start_time = time.time()
    
    if not request.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Retrieve relevant documents
        retrieval_start = time.time()
        results = retrieve_documents(request.q, top_k=request.top_k)
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


@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base."""
    try:
        from .retrieval import get_retriever
        retriever = get_retriever()
        
        stats = {}
        
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Documentation Copilot",
        "description": "AI assistant for ML infrastructure documentation",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask questions about ML documentation",
            "/health": "GET - Health check",
            "/stats": "GET - Knowledge base statistics",
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

