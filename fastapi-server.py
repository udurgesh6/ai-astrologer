"""
FastAPI Server for Astrology Q&A
Converts the Streamlit app to a REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')
sys.path.insert(0, 'src/agent')

from complete_pipeline2 import CompletePipeline2
from domain_definitions import DOMAIN_DEFINITIONS
from orchestration_agent2 import OrchestrationAgent2


# Global state to hold pipeline and agent
class AppState:
    pipeline: Optional[CompletePipeline2] = None
    agent: Optional[OrchestrationAgent2] = None
    initialized: bool = False
    error_message: Optional[str] = None


state = AppState()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize pipeline and agent
    print("🔄 Loading knowledge base...")
    try:
        state.pipeline = CompletePipeline2(
            base_dir="data2",
            chunk_size=800,
            chunk_overlap=150,
            use_llm_extraction=False,
            rerank_method="simple"
        )
        
        state.agent = OrchestrationAgent2()
        
        available_domains = state.pipeline.list_domains()
        
        if not available_domains:
            state.error_message = "No domains found in knowledge base"
            print(f"❌ {state.error_message}")
        else:
            state.initialized = True
            print(f"✅ Knowledge base loaded with {len(available_domains)} domains")
    
    except Exception as e:
        state.error_message = f"Initialization error: {str(e)}"
        print(f"❌ {state.error_message}")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Astrology Q&A API",
    description="AI-powered astrology question answering system",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    domains: Optional[List[str]] = None


class Source(BaseModel):
    source: str
    page: int
    content_preview: str
    domain: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]
    domains_searched: List[str]
    confidence: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    error_message: Optional[str] = None
    available_domains: Optional[List[str]] = None
    total_documents: Optional[int] = None


class DomainInfo(BaseModel):
    name: str
    description: str
    always_include: bool
    vector_count: Optional[int] = None


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Astrology Q&A API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    response = {
        "status": "healthy" if state.initialized else "unhealthy",
        "initialized": state.initialized,
        "error_message": state.error_message
    }
    
    if state.initialized:
        try:
            available_domains = state.pipeline.list_domains()
            stats = state.pipeline.get_statistics()
            total_vectors = sum(d.get('vector_count', 0) for d in stats['domains'].values())
            
            response["available_domains"] = available_domains
            response["total_documents"] = total_vectors
        except Exception as e:
            response["error_message"] = f"Error fetching stats: {str(e)}"
    
    return response


@app.get("/domains", response_model=List[DomainInfo])
async def list_domains():
    """List all available domains"""
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail=f"Service not initialized: {state.error_message}"
        )
    
    try:
        available_domains = state.pipeline.list_domains()
        stats = state.pipeline.get_statistics()
        
        domain_list = []
        for domain in available_domains:
            domain_info = DOMAIN_DEFINITIONS.get(domain, {})
            domain_list.append(DomainInfo(
                name=domain,
                description=domain_info.get('description', 'No description'),
                always_include=domain_info.get('always_include', False),
                vector_count=stats['domains'].get(domain, {}).get('vector_count', 0)
            ))
        
        return domain_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing domains: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint to ask a question
    
    - **question**: The question to ask
    - **top_k**: Number of top results to retrieve (default: 10)
    - **domains**: Optional list of specific domains to search (if None, agent will route automatically)
    """
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail=f"Service not initialized: {state.error_message}"
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Retrieval function for agent
        def retrieve_chunks(q, domains=None):
            return state.pipeline.search(
                q,
                top_k=request.top_k,
                domains=domains or request.domains,
                use_vector=True,
                use_keyword=True,
                use_graph=True,
                rerank=True
            )
        
        # Get available domains
        available_domains = state.pipeline.list_domains()
        
        # Process query through agent
        result = state.agent.process_query(
            request.question,
            retrieve_chunks,
            available_domains=available_domains
        )
        
        # Format sources
        formatted_sources = []
        for source in result.get('sources', []):
            formatted_sources.append(Source(
                source=source.get('source', 'Unknown'),
                page=source.get('page', 0),
                content_preview=source.get('content_preview', '')[:200],
                domain=source.get('domain')
            ))
        
        return AnswerResponse(
            answer=result['answer'],
            sources=formatted_sources,
            domains_searched=result.get('suggested_domains', []),
            confidence=result.get('confidence')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)