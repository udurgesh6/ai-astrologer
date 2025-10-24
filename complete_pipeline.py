"""
Complete PDF Processing Pipeline - Sprint 3
Integrates all features: Vector + Keyword + Graph + Reranking
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import time
import sys

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')

from pdf_extractor import PDFExtractor
from text_chunker import TextChunker
from vector_store import VectorStore
from entity_extractor import HybridEntityExtractor
from knowledge_graph import KnowledgeGraph
from graph_retrieval import GraphRetrieval
from keyword_search import KeywordSearch
from hybrid_retrieval import HybridRetrieval
from reranker import Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """
    Complete pipeline with all Sprint 1, 2, and 3 features:
    - PDF extraction and chunking
    - Entity extraction and knowledge graph
    - Vector search (semantic)
    - Keyword search (BM25)
    - Graph-based retrieval
    - Hybrid fusion (RRF)
    - Reranking
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        use_llm_extraction: bool = False,
        rerank_method: str = "simple"
    ):
        """
        Initialize complete pipeline
        
        Args:
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            use_llm_extraction: Use LLM for entity extraction
            rerank_method: Reranking method ("simple", "cross-encoder", "cohere")
        """
        # Core components
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self.keyword_search = KeywordSearch()
        
        # Processing
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.entity_extractor = HybridEntityExtractor(use_llm=use_llm_extraction)
        
        # Retrieval
        self.graph_retrieval = GraphRetrieval(
            self.knowledge_graph,
            self.entity_extractor
        )
        
        self.hybrid_retrieval = HybridRetrieval(
            vector_store=self.vector_store,
            keyword_search=self.keyword_search,
            graph_retrieval=self.graph_retrieval
        )
        
        # Reranking
        self.reranker = Reranker(method=rerank_method)
        
        logger.info(f"Complete pipeline initialized (reranking: {rerank_method})")
    
    def process_pdf(
        self,
        pdf_path: str,
        replace_existing: bool = True,
        extract_entities: bool = True
    ) -> Dict:
        """
        Process PDF through complete pipeline
        
        Args:
            pdf_path: Path to PDF
            replace_existing: Replace existing data
            extract_entities: Extract entities and build graph
            
        Returns:
            Processing results
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        logger.info(f"="*60)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"="*60)
        
        try:
            # Step 1: Extract PDF
            logger.info("Step 1/7: Extracting text from PDF...")
            if not PDFExtractor.validate_pdf(str(pdf_path)):
                raise ValueError(f"Invalid PDF: {pdf_path}")
            
            extractor = PDFExtractor(str(pdf_path))
            extraction_result = extractor.extract_text()
            pages = extraction_result['pages']
            logger.info(f"✓ Extracted {len(pages)} pages")
            
            # Step 2: Chunk text
            logger.info("Step 2/7: Chunking text...")
            chunks = self.chunker.chunk_document(
                pages=pages,
                source=pdf_path.name,
                strategy="paragraphs"
            )
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            # Step 3: Extract entities
            entity_stats = {'total_entities': 0, 'total_relationships': 0}
            if extract_entities:
                logger.info("Step 3/7: Extracting entities and building knowledge graph...")
                entity_stats = self._process_entities(chunks, pdf_path.name)
                logger.info(f"✓ Extracted {entity_stats['total_entities']} entities, "
                          f"{entity_stats['total_relationships']} relationships")
            else:
                logger.info("Step 3/7: Skipping entity extraction")
            
            # Step 4: Remove existing data
            if replace_existing:
                logger.info("Step 4/7: Removing existing data...")
                self.vector_store.delete_by_source(pdf_path.name)
            
            # Step 5: Build keyword index (NEW in Sprint 3)
            logger.info("Step 5/7: Building keyword search index...")
            self.keyword_search.index_documents(chunks)
            keyword_stats = self.keyword_search.get_statistics()
            logger.info(f"✓ Indexed {keyword_stats['total_documents']} documents, "
                       f"vocab size: {keyword_stats['vocabulary_size']}")
            
            # Step 6: Add to vector store
            logger.info("Step 6/7: Adding chunks to vector store...")
            self.vector_store.add_documents(chunks)
            logger.info(f"✓ Added {len(chunks)} chunks to vector store")
            
            # Step 7: Save indexes
            logger.info("Step 7/7: Saving indexes...")
            if extract_entities:
                graph_path = f"data/graphs/{pdf_path.stem}_graph.pkl"
                self.knowledge_graph.save_graph(graph_path)
            
            keyword_path = f"data/indexes/{pdf_path.stem}_keyword.pkl"
            Path(keyword_path).parent.mkdir(parents=True, exist_ok=True)
            self.keyword_search.save_index(keyword_path)
            logger.info(f"✓ Saved indexes")
            
            # Calculate results
            end_time = time.time()
            processing_time = end_time - start_time
            
            results = {
                'success': True,
                'source': pdf_path.name,
                'total_pages': len(pages),
                'total_chunks': len(chunks),
                'total_entities': entity_stats['total_entities'],
                'total_relationships': entity_stats['total_relationships'],
                'graph_nodes': self.knowledge_graph.graph.number_of_nodes(),
                'graph_edges': self.knowledge_graph.graph.number_of_edges(),
                'keyword_vocab_size': keyword_stats['vocabulary_size'],
                'processing_time_seconds': processing_time,
                'metadata': extraction_result['metadata']
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ Processing Complete!")
            logger.info(f"{'='*60}")
            logger.info(f"Pages: {len(pages)}")
            logger.info(f"Chunks: {len(chunks)}")
            logger.info(f"Entities: {entity_stats['total_entities']}")
            logger.info(f"Graph: {results['graph_nodes']} nodes, {results['graph_edges']} edges")
            logger.info(f"Keyword vocab: {keyword_stats['vocabulary_size']} terms")
            logger.info(f"Time: {processing_time:.2f}s")
            logger.info(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'source': pdf_path.name,
                'error': str(e)
            }
    
    def _process_entities(self, chunks, source: str) -> Dict:
        """Extract entities and build knowledge graph"""
        total_entities = 0
        total_relationships = 0
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            
            extraction = self.entity_extractor.extract_from_chunk(chunk, include_relationships=True)
            self.knowledge_graph.add_from_extraction(extraction, chunk.chunk_id)
            self.entity_extractor.tag_chunk_with_entities(chunk)
            
            total_entities += extraction['entity_count']
            total_relationships += extraction['relationship_count']
        
        return {
            'total_entities': total_entities,
            'total_relationships': total_relationships
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_vector: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
        rerank: bool = True,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        Complete search pipeline (NEW in Sprint 3)
        
        Args:
            query: Search query
            top_k: Number of results
            use_vector: Use vector search
            use_keyword: Use keyword search
            use_graph: Use graph retrieval
            rerank: Apply reranking
            fusion_method: Fusion method ("rrf" or "weighted")
            
        Returns:
            List of ranked results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Search Query: '{query}'")
        logger.info(f"{'='*60}")
        
        # Step 1: Hybrid retrieval
        logger.info("Retrieving with hybrid methods...")
        results = self.hybrid_retrieval.retrieve(
            query=query,
            top_k=top_k * 2 if rerank else top_k,  # Get more if reranking
            use_vector=use_vector,
            use_keyword=use_keyword,
            use_graph=use_graph,
            fusion_method=fusion_method
        )
        
        logger.info(f"Retrieved {len(results)} candidates")
        
        # Step 2: Rerank (NEW in Sprint 3)
        if rerank and results:
            logger.info("Reranking results...")
            results = self.reranker.rerank(query, results, top_k=top_k)
            logger.info(f"Reranked to top {len(results)} results")
        
        logger.info(f"{'='*60}\n")
        return results
    
    def get_related_concepts(self, query: str, max_concepts: int = 10) -> List[Dict]:
        """Get related astrological concepts"""
        return self.graph_retrieval.get_related_concepts(query, max_concepts)
    
    def explain_entity_connection(self, entity1: str, entity2: str) -> Optional[Dict]:
        """Explain connection between entities"""
        return self.graph_retrieval.explain_connection(entity1, entity2)
    
    def get_statistics(self) -> Dict:
        """Get complete system statistics"""
        return {
            'vector_store': self.vector_store.get_collection_stats(),
            'knowledge_graph': self.knowledge_graph.get_statistics(),
            'keyword_search': self.keyword_search.get_statistics(),
            'hybrid_retrieval': self.hybrid_retrieval.get_statistics()
        }


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        exit(1)
    
    print("="*60)
    print("Complete Pipeline - Sprint 3")
    print("Vector + Keyword + Graph + Reranking")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        chunk_size=800,
        chunk_overlap=150,
        use_llm_extraction=False,
        rerank_method="simple"  # or "cross-encoder" if installed
    )
    
    # Process PDF
    pdf_path = "data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf"
    
    if Path(pdf_path).exists():
        result = pipeline.process_pdf(pdf_path, extract_entities=True)
        
        if result['success']:
            # Test complete search
            print("\n" + "="*60)
            print("Testing Complete Search Pipeline")
            print("="*60)
            
            queries = [
                "Mars in 10th house career",
                "marriage timing",
                "wealth indicators"
            ]
            
            for query in queries:
                print(f"\n{'='*60}")
                print(f"Query: '{query}'")
                print(f"{'='*60}")
                
                results = pipeline.search(
                    query=query,
                    top_k=3,
                    use_vector=True,
                    use_keyword=True,
                    use_graph=True,
                    rerank=True
                )
                
                for i, r in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"  Score: {r.get('rerank_score', r.get('fusion_score', 0)):.3f}")
                    print(f"  Sources: {', '.join(r['sources'])}")
                    print(f"  Page: {r['metadata'].get('page_number', '?')}")
                    print(f"  Content: {r['content'][:150]}...")
            
            # System statistics
            print("\n" + "="*60)
            print("System Statistics")
            print("="*60)
            
            stats = pipeline.get_statistics()
            print(f"\nVector Store: {stats['vector_store']['total_documents']} documents")
            print(f"Knowledge Graph: {stats['knowledge_graph']['total_nodes']} nodes, "
                  f"{stats['knowledge_graph']['total_edges']} edges")
            print(f"Keyword Index: {stats['keyword_search']['vocabulary_size']} unique terms")
            print(f"\nAll retrieval methods available: "
                  f"Vector={stats['hybrid_retrieval']['vector_available']}, "
                  f"Keyword={stats['hybrid_retrieval']['keyword_available']}, "
                  f"Graph={stats['hybrid_retrieval']['graph_available']}")
        
        else:
            print(f"\n✗ Processing failed: {result.get('error')}")
    
    else:
        print(f"\nERROR: PDF not found at {pdf_path}")
        print("Please add a PDF to test the complete pipeline")