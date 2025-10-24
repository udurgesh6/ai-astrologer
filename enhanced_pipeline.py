"""
Enhanced PDF Processing Pipeline
Includes entity extraction and knowledge graph construction (Sprint 2)
"""

from typing import List
import logging
from pathlib import Path
from typing import Optional, Dict
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPDFPipeline:
    """
    Complete pipeline with knowledge graph support
    PDF → Text → Chunks → Entities → Graph + Vector Store
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        use_llm_extraction: bool = False
    ):
        """
        Initialize enhanced pipeline
        
        Args:
            vector_store: VectorStore instance
            knowledge_graph: KnowledgeGraph instance
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            use_llm_extraction: Whether to use LLM for entity extraction
        """
        self.vector_store = vector_store or VectorStore()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.entity_extractor = HybridEntityExtractor(use_llm=use_llm_extraction)
        
        self.graph_retrieval = GraphRetrieval(
            self.knowledge_graph,
            self.entity_extractor
        )
        
        logger.info("Enhanced PDF Pipeline initialized")
    
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
            extract_entities: Whether to extract entities and build graph
            
        Returns:
            Processing results
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting enhanced processing: {pdf_path.name}")
        
        try:
            # Step 1: Extract PDF text
            logger.info("Step 1/6: Extracting text from PDF...")
            if not PDFExtractor.validate_pdf(str(pdf_path)):
                raise ValueError(f"Invalid PDF: {pdf_path}")
            
            extractor = PDFExtractor(str(pdf_path))
            extraction_result = extractor.extract_text()
            pages = extraction_result['pages']
            
            logger.info(f"Extracted {len(pages)} pages")
            
            # Step 2: Chunk text
            logger.info("Step 2/6: Chunking text...")
            chunks = self.chunker.chunk_document(
                pages=pages,
                source=pdf_path.name,
                strategy="paragraphs"
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Extract entities (NEW in Sprint 2)
            entity_stats = {'total_entities': 0, 'total_relationships': 0}
            
            if extract_entities:
                logger.info("Step 3/6: Extracting entities and building knowledge graph...")
                entity_stats = self._process_entities(chunks, pdf_path.name)
                logger.info(f"Extracted {entity_stats['total_entities']} entities, "
                          f"{entity_stats['total_relationships']} relationships")
            else:
                logger.info("Step 3/6: Skipping entity extraction")
            
            # Step 4: Remove existing data
            if replace_existing:
                logger.info("Step 4/6: Removing existing data...")
                self.vector_store.delete_by_source(pdf_path.name)
            
            # Step 5: Add to vector store
            logger.info("Step 5/6: Adding chunks to vector store...")
            self.vector_store.add_documents(chunks)
            
            # Step 6: Save knowledge graph
            if extract_entities:
                logger.info("Step 6/6: Saving knowledge graph...")
                graph_path = f"data/graphs/{pdf_path.stem}_graph.pkl"
                self.knowledge_graph.save_graph(graph_path)
            
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
                'processing_time_seconds': processing_time,
                'metadata': extraction_result['metadata']
            }
            
            logger.info(f"\n✓ Successfully processed {pdf_path.name}")
            logger.info(f"  Pages: {len(pages)}")
            logger.info(f"  Chunks: {len(chunks)}")
            logger.info(f"  Entities: {entity_stats['total_entities']}")
            logger.info(f"  Graph nodes: {results['graph_nodes']}")
            logger.info(f"  Time: {processing_time:.2f}s")
            
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
        """
        Extract entities from chunks and build knowledge graph
        
        Args:
            chunks: List of Chunk objects
            source: Source document name
            
        Returns:
            Dictionary with entity statistics
        """
        total_entities = 0
        total_relationships = 0
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract entities and relationships
            extraction = self.entity_extractor.extract_from_chunk(
                chunk,
                include_relationships=True
            )
            
            # Add to knowledge graph
            self.knowledge_graph.add_from_extraction(extraction, chunk.chunk_id)
            
            # Tag chunk with entities
            self.entity_extractor.tag_chunk_with_entities(chunk)
            
            total_entities += extraction['entity_count']
            total_relationships += extraction['relationship_count']
        
        return {
            'total_entities': total_entities,
            'total_relationships': total_relationships
        }
    
    def hybrid_search(
        self,
        query: str,
        use_vector: bool = True,
        use_graph: bool = True,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search using both vector similarity and graph traversal
        
        Args:
            query: Search query
            use_vector: Use vector similarity search
            use_graph: Use graph-based retrieval
            top_k: Number of results
            
        Returns:
            List of result dictionaries
        """
        all_chunk_ids = set()
        
        # Vector search
        if use_vector:
            vector_results = self.vector_store.similarity_search(query, k=top_k)
            vector_chunk_ids = {r['id'] for r in vector_results}
            all_chunk_ids.update(vector_chunk_ids)
            logger.debug(f"Vector search found {len(vector_chunk_ids)} chunks")
        
        # Graph search
        if use_graph:
            graph_chunk_ids = self.graph_retrieval.retrieve(query, max_hops=2, top_k=top_k)
            all_chunk_ids.update(graph_chunk_ids)
            logger.debug(f"Graph search found {len(graph_chunk_ids)} chunks")
        
        # Get chunk content for all IDs
        results = []
        for chunk_id in list(all_chunk_ids)[:top_k]:
            # Fetch from vector store
            vector_results = self.vector_store.collection.get(ids=[chunk_id])
            
            if vector_results['documents']:
                results.append({
                    'id': chunk_id,
                    'content': vector_results['documents'][0],
                    'metadata': vector_results['metadatas'][0] if vector_results['metadatas'] else {}
                })
        
        logger.info(f"Hybrid search returned {len(results)} results")
        return results
    
    def get_related_concepts(self, query: str, max_concepts: int = 10) -> List[Dict]:
        """Get related astrological concepts for a query"""
        return self.graph_retrieval.get_related_concepts(query, max_concepts)
    
    def explain_entity_connection(self, entity1: str, entity2: str) -> Optional[Dict]:
        """Explain connection between two entities"""
        return self.graph_retrieval.explain_connection(entity1, entity2)
    
    def get_graph_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        return self.knowledge_graph.get_statistics()
    
    def visualize_graph(self, output_file: str = "knowledge_graph.png") -> None:
        """Create graph visualization"""
        self.knowledge_graph.visualize(output_file)


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
    print("Enhanced PDF Pipeline with Knowledge Graph")
    print("="*60)
    
    # Initialize pipeline
    pipeline = EnhancedPDFPipeline(
        chunk_size=800,
        chunk_overlap=150,
        use_llm_extraction=False  # Set to True if you have Anthropic key
    )
    
    # Process PDF
    pdf_path = "data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf"
    
    if Path(pdf_path).exists():
        print(f"\nProcessing: {pdf_path}")
        
        result = pipeline.process_pdf(
            pdf_path,
            extract_entities=True  # NEW: Extract entities and build graph
        )
        
        if result['success']:
            print("\n✓ Processing complete!")
            print(f"  Pages: {result['total_pages']}")
            print(f"  Chunks: {result['total_chunks']}")
            print(f"  Entities extracted: {result['total_entities']}")
            print(f"  Relationships found: {result['total_relationships']}")
            print(f"  Graph nodes: {result['graph_nodes']}")
            print(f"  Graph edges: {result['graph_edges']}")
            print(f"  Time: {result['processing_time_seconds']:.2f}s")
            
            # Test hybrid search
            print("\n" + "="*60)
            print("Testing Hybrid Search (Vector + Graph)")
            print("="*60)
            
            query = "What about Mars and career?"
            print(f"\nQuery: {query}")
            
            results = pipeline.hybrid_search(query, top_k=3)
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Page: {result['metadata'].get('page_number', 'Unknown')}")
                print(f"Content: {result['content'][:200]}...")
            
            # Test related concepts
            print("\n" + "="*60)
            print("Related Concepts")
            print("="*60)
            
            related = pipeline.get_related_concepts(query, max_concepts=5)
            print(f"\nRelated to '{query}':")
            for concept in related:
                print(f"  - {concept['entity']} ({concept['type']}) - {concept['distance']} hops from {concept['from']}")
            
            # Graph statistics
            print("\n" + "="*60)
            print("Knowledge Graph Statistics")
            print("="*60)
            
            stats = pipeline.get_graph_statistics()
            print(f"\nGraph Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Visualize graph
            print("\nCreating graph visualization...")
            try:
                pipeline.visualize_graph("knowledge_graph.png")
                print("✓ Graph visualization saved to knowledge_graph.png")
            except:
                print("⚠ Could not create visualization (matplotlib may not be installed)")
        
        else:
            print(f"\n✗ Processing failed: {result.get('error')}")
    
    else:
        print(f"\nERROR: PDF not found at {pdf_path}")
        print("Please add a PDF to test the enhanced pipeline")