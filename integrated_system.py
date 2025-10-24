"""
Integrated System - Sprint 4
Complete system with agent orchestration and Q&A capability
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import sys

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')
sys.path.insert(0, 'src/agent')

from complete_pipeline import CompletePipeline
from orchestration_agent import OrchestrationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedSystem:
    """
    Complete integrated system combining:
    - PDF processing (Sprints 1-3)
    - Orchestration agent (Sprint 4 with OpenAI)
    - Query understanding and answer synthesis
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chunk_size: int = 800,
        rerank_method: str = "simple"
    ):
        """
        Initialize integrated system
        
        Args:
            openai_api_key: API key for OpenAI (optional, uses env)
            chunk_size: Chunk size for processing
            rerank_method: Reranking method
        """
        logger.info("Initializing integrated system...")
        
        # Initialize pipeline (all retrieval methods)
        self.pipeline = CompletePipeline(
            chunk_size=chunk_size,
            chunk_overlap=150,
            use_llm_extraction=False,
            rerank_method=rerank_method
        )
        
        # Initialize orchestration agent with OpenAI
        self.agent = OrchestrationAgent(api_key=openai_api_key)
        
        logger.info("✓ Integrated system ready")
    
    def process_document(
        self,
        pdf_path: str,
        replace_existing: bool = True
    ) -> Dict:
        """
        Process a PDF document
        
        Args:
            pdf_path: Path to PDF
            replace_existing: Replace if exists
            
        Returns:
            Processing result
        """
        return self.pipeline.process_pdf(
            pdf_path,
            replace_existing=replace_existing,
            extract_entities=True
        )
    
    def ask(
        self,
        question: str,
        verbose: bool = False
    ) -> Dict:
        """
        Ask a question and get comprehensive answer
        
        Args:
            question: User question
            verbose: Show detailed processing info
            
        Returns:
            Answer dictionary with sources
        """
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Question: {question}")
            logger.info(f"{'='*60}")
        
        # Define retrieval function for agent
        def retrieve(query):
            if verbose:
                logger.info(f"Retrieving for: {query}")
            
            results = self.pipeline.search(
                query=query,
                top_k=10,
                use_vector=True,
                use_keyword=True,
                use_graph=True,
                rerank=True
            )
            
            if verbose:
                logger.info(f"Found {len(results)} results")
            
            return results
        
        # Process through agent
        result = self.agent.process_query(
            question,
            retrieve
        )
        
        if verbose:
            logger.info(f"\nAnswer generated:")
            logger.info(f"- Type: {result['query_type']}")
            logger.info(f"- Sources: {result['num_sources']}")
            logger.info(f"- Confidence: {result['confidence']}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return self.pipeline.get_statistics()


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        exit(1)
    
    print("="*60)
    print("Integrated System - Sprint 4")
    print("Complete Q&A System with Intelligent Agent (OpenAI)")
    print("="*60)
    
    # Initialize system
    print("\nInitializing system...")
    system = IntegratedSystem()
    print("✓ System initialized")
    
    # Process PDF if exists
    pdf_path = "data/pdfs/sample_astrology.pdf"
    
    if Path(pdf_path).exists():
        print(f"\nProcessing: {pdf_path}")
        result = system.process_document(pdf_path)
        
        if result['success']:
            print(f"✓ PDF processed successfully")
            print(f"  - {result['total_chunks']} chunks")
            print(f"  - {result['total_entities']} entities")
            print(f"  - {result['graph_nodes']} graph nodes")
            
            # Test Q&A
            print("\n" + "="*60)
            print("Testing Intelligent Q&A")
            print("="*60)
            
            test_questions = [
                "What determines career success in astrology?",
                "I have Mars in 10th house, what about my career?",
                "What happens during Sun Mahadasha?"
            ]
            
            for question in test_questions:
                print(f"\n{'='*60}")
                print(f"Q: {question}")
                print(f"{'='*60}")
                
                answer = system.ask(question, verbose=True)
                
                print(f"\nA: {answer['answer'][:500]}...")
                
                if answer['sources']:
                    print(f"\nSources ({len(answer['sources'])}):")
                    for i, source in enumerate(answer['sources'][:3]):
                        print(f"  {i+1}. Page {source['page']}: {source['content_preview'][:100]}...")
                
                print()
                input("Press Enter to continue...")
            
            # Show statistics
            print("\n" + "="*60)
            print("System Statistics")
            print("="*60)
            
            stats = system.get_statistics()
            print(f"\nVector Store: {stats['vector_store']['total_documents']} documents")
            print(f"Knowledge Graph: {stats['knowledge_graph']['total_nodes']} nodes")
            print(f"Keyword Index: {stats['keyword_search']['vocabulary_size']} terms")
            
        else:
            print(f"✗ Processing failed: {result.get('error')}")
    
    else:
        print(f"\nERROR: PDF not found at {pdf_path}")
        print("Please add a PDF to test the integrated system")
        print("\nYou can still test with an existing processed document:")
        print("from integrated_system import IntegratedSystem")
        print("system = IntegratedSystem()")
        print('answer = system.ask("What determines career success?")')
        print("print(answer['answer'])")