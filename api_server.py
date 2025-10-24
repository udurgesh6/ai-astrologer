"""
FastAPI Server for Astrology PDF Q&A
Serves the Next.js frontend with API endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
from typing import Optional
import uvicorn

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')
sys.path.insert(0, 'src/agent')

from complete_pipeline import CompletePipeline
from orchestration_agent import OrchestrationAgent

app = FastAPI(title="Astrology PDF Q&A API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper state management/database)
pipeline: Optional[CompletePipeline] = None
agent: Optional[OrchestrationAgent] = None
pdf_processed = False


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    num_sources: int
    confidence: str
    sources: list


@app.get("/")
async def root():
    return {"message": "Astrology PDF Q&A API", "status": "running"}


@app.post("/api/initialize")
async def initialize_system():
    """Initialize the pipeline and agent"""
    global pipeline, agent
    
    try:
        pipeline = CompletePipeline(
            chunk_size=800,
            chunk_overlap=150,
            use_llm_extraction=False,
            rerank_method="simple"
        )
        agent = OrchestrationAgent()
        
        return {
            "success": True,
            "message": "System initialized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "initialized": pipeline is not None,
        "pdf_processed": pdf_processed
    }


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Process uploaded PDF"""
    global pdf_processed
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        pdf_path = Path("data/pdfs") / file.filename
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        # Process PDF
        result = pipeline.process_pdf(
            str(pdf_path),
            extract_entities=True
        )
        
        if result['success']:
            pdf_processed = True
            return {
                "success": True,
                "message": "PDF processed successfully",
                "stats": {
                    "pages": result['total_pages'],
                    "chunks": result['total_chunks'],
                    "entities": result['total_entities'],
                    "graph_nodes": result['graph_nodes'],
                    "processing_time": f"{result['processing_time_seconds']:.1f}s"
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and return answer with sources"""
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not pdf_processed:
        raise HTTPException(status_code=400, detail="No PDF processed yet")
    
    try:
        # Retrieval function for agent
        def retrieve_chunks(q):
            return pipeline.search(
                q,
                top_k=10,
                use_vector=True,
                use_keyword=True,
                use_graph=True,
                rerank=True
            )
        
        # Process query through agent
        result = agent.process_query(
            request.query,
            retrieve_chunks
        )
        
        return QueryResponse(
            answer=result['answer'],
            query_type=result.get('query_type', 'Unknown'),
            num_sources=result.get('num_sources', 0),
            confidence=result.get('confidence', 'unknown'),
            sources=result.get('sources', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not pdf_processed:
        return {
            "vector_store": {"total_documents": 0},
            "knowledge_graph": {"total_nodes": 0},
            "keyword_search": {"vocabulary_size": 0}
        }
    
    try:
        stats = pipeline.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)