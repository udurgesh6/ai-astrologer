"""
Complete PDF Processing Pipeline v3 - Domain-Based Knowledge Bases
Intelligent routing to domain-specific KBs with base knowledge
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Set
import time
import sys
import json
from datetime import datetime

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
from domain_definitions import DOMAIN_DEFINITIONS, get_all_domains, get_always_include_domains

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainKnowledgeBase:
    """Represents a domain-specific knowledge base"""
    
    def __init__(self, domain_name: str, base_dir: Path):
        self.domain_name = domain_name
        self.base_dir = base_dir / domain_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.graph_path = self.base_dir / 'graph.pkl'
        self.keyword_path = self.base_dir / 'keyword.pkl'
        self.vector_db_path = self.base_dir / 'vector_db'
        self.metadata_path = self.base_dir / 'metadata.json'
        
        # Components
        self.vector_store = VectorStore(persist_directory=str(self.vector_db_path))
        self.knowledge_graph = KnowledgeGraph()
        self.keyword_search = KeywordSearch()
        self.processed_files: Set[str] = set()
        
        # Load existing data if available
        self._load_existing()
    
    def _load_existing(self):
        """Load existing KB data"""
        if self.graph_path.exists():
            self.knowledge_graph.load_graph(str(self.graph_path))
            logger.info(f"  Loaded existing graph for {self.domain_name}")
        
        if self.keyword_path.exists():
            self.keyword_search.load_index(str(self.keyword_path))
            logger.info(f"  Loaded existing keyword index for {self.domain_name}")
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.processed_files = set(metadata.get('processed_files', []))
                logger.info(f"  Found {len(self.processed_files)} processed files in {self.domain_name}")
    
    def save(self):
        """Save KB data"""
        self.knowledge_graph.save_graph(str(self.graph_path))
        self.keyword_search.save_index(str(self.keyword_path))
        
        # Save metadata with safe key access
        try:
            graph_stats = self.knowledge_graph.get_statistics()
            keyword_stats = self.keyword_search.get_statistics()
            vector_stats = self.vector_store.get_collection_stats()
            
            metadata = {
                'domain': self.domain_name,
                'processed_files': list(self.processed_files),
                'last_updated': datetime.now().isoformat(),
                'stats': {
                    'graph_nodes': graph_stats.get('total_nodes', 0),
                    'graph_edges': graph_stats.get('total_edges', 0),
                    'keyword_vocab': keyword_stats.get('vocabulary_size', 0),
                    'vector_count': vector_stats.get('total_documents', 0)
                }
            }
        except Exception as e:
            logger.warning(f"Error getting stats for metadata: {e}")
            # Fallback to basic metadata
            metadata = {
                'domain': self.domain_name,
                'processed_files': list(self.processed_files),
                'last_updated': datetime.now().isoformat(),
                'stats': {
                    'graph_nodes': 0,
                    'graph_edges': 0,
                    'keyword_vocab': 0,
                    'vector_count': 0
                }
            }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_processed(self, pdf_name: str) -> bool:
        """Check if PDF already processed"""
        return pdf_name in self.processed_files
    
    def add_processed_file(self, pdf_name: str):
        """Mark file as processed"""
        self.processed_files.add(pdf_name)


class CompletePipeline2:
    """
    Domain-based pipeline with intelligent routing:
    1. Separate KB for each domain (children, marriage, remedies, etc.)
    2. Base KB (general astrology) always included
    3. Incremental processing (skip already processed PDFs)
    4. Intelligent query routing to appropriate domain(s)
    """
    
    def __init__(
        self,
        base_dir: str = "data2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        use_llm_extraction: bool = False,
        rerank_method: str = "simple"
    ):
        """
        Initialize domain-based pipeline
        
        Args:
            base_dir: Base directory for all domain KBs
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            use_llm_extraction: Use LLM for entity extraction
            rerank_method: Reranking method
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing components
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.entity_extractor = HybridEntityExtractor(use_llm=use_llm_extraction)
        self.reranker = Reranker(method=rerank_method)
        
        # Domain knowledge bases
        self.domain_kbs: Dict[str, DomainKnowledgeBase] = {}
        
        # PDF to domain mapping (will be loaded/saved)
        self.pdf_domain_map: Dict[str, str] = {}
        self.map_file = self.base_dir / 'pdf_domain_map.json'
        self._load_pdf_domain_map()
        
        # Auto-discover and load existing domains
        self._discover_existing_domains()
        
        logger.info(f"="*60)
        logger.info(f"Domain-Based Pipeline v2 Initialized")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Loaded {len(self.domain_kbs)} existing domains: {list(self.domain_kbs.keys())}")
        logger.info(f"="*60)
    
    def _discover_existing_domains(self):
        """Auto-discover and load existing domain directories"""
        if not self.base_dir.exists():
            return
        
        # Look for domain directories
        for item in self.base_dir.iterdir():
            if item.is_dir():
                domain_name = item.name
                # Check if it looks like a domain KB (has metadata.json)
                metadata_file = item / 'metadata.json'
                if metadata_file.exists():
                    try:
                        # Load this domain KB
                        kb = DomainKnowledgeBase(domain_name, self.base_dir)
                        self.domain_kbs[domain_name] = kb
                        logger.info(f"  Loaded domain: {domain_name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load domain {domain_name}: {e}")
    
    
    def _load_pdf_domain_map(self):
        """Load PDF to domain mapping"""
        if self.map_file.exists():
            with open(self.map_file, 'r') as f:
                self.pdf_domain_map = json.load(f)
                logger.info(f"Loaded mapping for {len(self.pdf_domain_map)} PDFs")
    
    def _save_pdf_domain_map(self):
        """Save PDF to domain mapping"""
        with open(self.map_file, 'w') as f:
            json.dump(self.pdf_domain_map, f, indent=2)
    
    def _get_or_create_kb(self, domain: str) -> DomainKnowledgeBase:
        """Get or create domain KB"""
        if domain not in self.domain_kbs:
            self.domain_kbs[domain] = DomainKnowledgeBase(domain, self.base_dir)
            logger.info(f"Created/Loaded KB for domain: {domain}")
        return self.domain_kbs[domain]
    
    def classify_pdf_domain(self, pdf_path: Path) -> str:
        """
        Classify PDF into a domain based on filename
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            Domain name
        """
        filename_lower = pdf_path.name.lower()
        
        # Check each domain's keywords
        for domain, info in DOMAIN_DEFINITIONS.items():
            if domain == 'base':
                continue
            
            for keyword in info['keywords']:
                if keyword.lower() in filename_lower:
                    logger.info(f"Classified '{pdf_path.name}' as domain: {domain}")
                    return domain
        
        # Default to 'general' if no match
        logger.info(f"Classified '{pdf_path.name}' as domain: general (no keywords matched)")
        return 'general'
    
    def process_pdfs(
        self,
        pdf_configs: List[Dict[str, str]],
        force_reprocess: bool = False
    ) -> Dict:
        """
        Process multiple PDFs into their respective domain KBs
        
        Args:
            pdf_configs: List of dicts with 'path' and optional 'domain'
                        Example: [
                            {'path': 'file1.pdf', 'domain': 'marriage'},
                            {'path': 'file2.pdf'}  # auto-classify
                        ]
            force_reprocess: Force reprocessing even if already processed
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DOMAIN-BASED MULTI-FILE PROCESSING")
        logger.info(f"Total PDFs: {len(pdf_configs)}")
        logger.info(f"Force reprocess: {force_reprocess}")
        logger.info(f"{'='*60}")
        
        results = {
            'processed': [],
            'skipped': [],
            'failed': [],
            'domains_updated': set()
        }
        
        # Step 1: Validate and classify PDFs
        logger.info(f"\nStep 1: Validating and classifying PDFs...")
        
        for config in pdf_configs:
            pdf_path = Path(config['path'])
            
            if not pdf_path.exists():
                logger.warning(f"  ✗ File not found: {pdf_path}")
                results['failed'].append({
                    'file': str(pdf_path),
                    'error': 'File not found'
                })
                continue
            
            if not PDFExtractor.validate_pdf(str(pdf_path)):
                logger.warning(f"  ✗ Invalid PDF: {pdf_path}")
                results['failed'].append({
                    'file': str(pdf_path),
                    'error': 'Invalid PDF'
                })
                continue
            
            # Determine domain
            if 'domain' in config:
                domain = config['domain']
                logger.info(f"  ✓ {pdf_path.name} → {domain} (manual)")
            else:
                domain = self.classify_pdf_domain(pdf_path)
                logger.info(f"  ✓ {pdf_path.name} → {domain} (auto)")
            
            # Get or create domain KB
            kb = self._get_or_create_kb(domain)
            
            # Check if already processed
            if not force_reprocess and kb.is_processed(pdf_path.name):
                logger.info(f"  ⊙ {pdf_path.name} already processed in {domain}, skipping")
                results['skipped'].append({
                    'file': pdf_path.name,
                    'domain': domain,
                    'reason': 'Already processed'
                })
                continue
            
            # Step 2: Process this PDF
            logger.info(f"\n{'-'*60}")
            logger.info(f"Processing: {pdf_path.name} → {domain}")
            logger.info(f"{'-'*60}")
            
            try:
                result = self._process_single_pdf(pdf_path, kb, domain)
                
                if result['success']:
                    kb.add_processed_file(pdf_path.name)
                    self.pdf_domain_map[pdf_path.name] = domain
                    results['processed'].append(result)
                    results['domains_updated'].add(domain)
                    logger.info(f"✓ Successfully processed {pdf_path.name}")
                else:
                    results['failed'].append({
                        'file': pdf_path.name,
                        'domain': domain,
                        'error': result.get('error')
                    })
                    logger.error(f"✗ Failed: {result.get('error')}")
            
            except Exception as e:
                logger.error(f"✗ Error processing {pdf_path.name}: {str(e)}")
                results['failed'].append({
                    'file': pdf_path.name,
                    'domain': domain,
                    'error': str(e)
                })
        
        # Step 3: Save all updated domain KBs
        logger.info(f"\n{'-'*60}")
        logger.info(f"Saving updated knowledge bases...")
        logger.info(f"{'-'*60}")
        
        for domain in results['domains_updated']:
            if domain in self.domain_kbs:
                self.domain_kbs[domain].save()
                logger.info(f"✓ Saved KB for domain: {domain}")
        
        # Save PDF mapping
        self._save_pdf_domain_map()
        logger.info(f"✓ Saved PDF-domain mapping")
        
        # Calculate summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        summary = {
            'success': len(results['failed']) == 0,
            'total_pdfs': len(pdf_configs),
            'processed': len(results['processed']),
            'skipped': len(results['skipped']),
            'failed': len(results['failed']),
            'domains_updated': list(results['domains_updated']),
            'processing_time_seconds': processing_time,
            'details': results
        }
        
        self._print_processing_summary(summary)
        
        return summary
    
    def _process_single_pdf(
        self,
        pdf_path: Path,
        kb: DomainKnowledgeBase,
        domain: str
    ) -> Dict:
        """Process single PDF into domain KB"""
        
        try:
            # Extract PDF
            logger.info(f"  Extracting text...")
            extractor = PDFExtractor(str(pdf_path))
            extraction_result = extractor.extract_text()
            pages = extraction_result['pages']
            logger.info(f"  ✓ Extracted {len(pages)} pages")
            
            # Chunk text
            logger.info(f"  Chunking text...")
            chunks = self.chunker.chunk_document(
                pages=pages,
                source=pdf_path.name,
                strategy="paragraphs"
            )
            logger.info(f"  ✓ Created {len(chunks)} chunks")
            
            # Extract entities
            logger.info(f"  Extracting entities...")
            entity_count = 0
            relationship_count = 0
            
            for chunk in chunks:
                extraction = self.entity_extractor.extract_from_chunk(
                    chunk,
                    include_relationships=True
                )
                kb.knowledge_graph.add_from_extraction(extraction, chunk.chunk_id)
                self.entity_extractor.tag_chunk_with_entities(chunk)
                
                entity_count += extraction['entity_count']
                relationship_count += extraction['relationship_count']
            
            logger.info(f"  ✓ Extracted {entity_count} entities, {relationship_count} relationships")
            
            # Add to keyword search
            logger.info(f"  Building keyword index...")
            kb.keyword_search.index_documents(chunks)
            logger.info(f"  ✓ Indexed {len(chunks)} documents")
            
            # Add to vector store
            logger.info(f"  Adding to vector store...")
            kb.vector_store.add_documents(chunks)
            logger.info(f"  ✓ Added {len(chunks)} vectors")
            
            return {
                'success': True,
                'file': pdf_path.name,
                'domain': domain,
                'pages': len(pages),
                'chunks': len(chunks),
                'entities': entity_count,
                'relationships': relationship_count
            }
            
        except Exception as e:
            logger.error(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'file': pdf_path.name,
                'domain': domain,
                'error': str(e)
            }
    
    def route_query(self, query: str) -> List[str]:
        """
        Route query to appropriate domains
        
        Args:
            query: User query
            
        Returns:
            List of domain names to search
        """
        query_lower = query.lower()
        relevant_domains = ['base']  # Always include base
        
        # Check each domain's keywords
        for domain, info in DOMAIN_DEFINITIONS.items():
            if domain == 'base':
                continue
            
            for keyword in info['keywords']:
                if keyword.lower() in query_lower:
                    relevant_domains.append(domain)
                    break
        
        # If no specific domain found, add 'general'
        if len(relevant_domains) == 1:  # Only 'base'
            if 'general' in self.domain_kbs:
                relevant_domains.append('general')
        
        logger.info(f"Query routed to domains: {relevant_domains}")
        return relevant_domains
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        domains: Optional[List[str]] = None,
        use_vector: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
        rerank: bool = True,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        Search across appropriate domain KBs
        
        Args:
            query: Search query
            top_k: Number of results
            domains: Specific domains to search (None = auto-route)
            use_vector: Use vector search
            use_keyword: Use keyword search
            use_graph: Use graph retrieval
            rerank: Apply reranking
            fusion_method: Fusion method
            
        Returns:
            List of ranked results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Search Query: '{query}'")
        logger.info(f"{'='*60}")
        
        # Route to domains
        if domains is None:
            domains = self.route_query(query)
        else:
            # Always include base
            if 'base' not in domains:
                domains = ['base'] + domains
        
        logger.info(f"Searching in domains: {domains}")
        
        # Collect results from each domain
        all_results = []
        seen_chunks = set()
        
        for domain in domains:
            if domain not in self.domain_kbs:
                logger.warning(f"Domain '{domain}' not found, skipping")
                continue
            
            kb = self.domain_kbs[domain]
            
            logger.info(f"\nSearching in {domain}...")
            
            # Create hybrid retrieval for this KB
            graph_retrieval = GraphRetrieval(
                kb.knowledge_graph,
                self.entity_extractor
            )
            
            hybrid_retrieval = HybridRetrieval(
                vector_store=kb.vector_store,
                keyword_search=kb.keyword_search,
                graph_retrieval=graph_retrieval
            )
            
            # Retrieve
            results = hybrid_retrieval.retrieve(
                query=query,
                top_k=top_k * 2 if rerank else top_k,
                use_vector=use_vector,
                use_keyword=use_keyword,
                use_graph=use_graph,
                fusion_method=fusion_method
            )
            
            # Add domain info and deduplicate
            for result in results:
                chunk_id = result.get('id')
                if chunk_id not in seen_chunks:
                    result['domain'] = domain
                    all_results.append(result)
                    seen_chunks.add(chunk_id)
            
            logger.info(f"  Retrieved {len(results)} from {domain}")
        
        logger.info(f"\nTotal unique results: {len(all_results)}")
        
        # Rerank across all domains
        if rerank and all_results:
            logger.info("Reranking results across all domains...")
            all_results = self.reranker.rerank(query, all_results, top_k=top_k)
            logger.info(f"Reranked to top {len(all_results)} results")
        else:
            # Just take top_k
            all_results = all_results[:top_k]
        
        logger.info(f"{'='*60}\n")
        return all_results
    
    def _print_processing_summary(self, summary: Dict):
        """Print processing summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ PROCESSING COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total PDFs: {summary['total_pdfs']}")
        logger.info(f"Processed: {summary['processed']}")
        logger.info(f"Skipped: {summary['skipped']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"{'='*60}")
        logger.info(f"Domains Updated: {', '.join(summary['domains_updated'])}")
        logger.info(f"Processing Time: {summary['processing_time_seconds']:.2f}s")
        logger.info(f"{'='*60}\n")
        
        if summary['skipped'] > 0:
            logger.info(f"Skipped files (already processed):")
            for item in summary['details']['skipped']:
                logger.info(f"  ⊙ {item['file']} ({item['domain']})")
        
        if summary['failed'] > 0:
            logger.warning(f"\nFailed files:")
            for item in summary['details']['failed']:
                logger.warning(f"  ✗ {item['file']}: {item.get('error', 'Unknown error')}")
    
    def get_statistics(self) -> Dict:
        """Get statistics for all domains"""
        stats = {
            'total_domains': len(self.domain_kbs),
            'domains': {}
        }
        
        for domain, kb in self.domain_kbs.items():
            try:
                vector_stats = kb.vector_store.get_collection_stats()
                graph_stats = kb.knowledge_graph.get_statistics()
                keyword_stats = kb.keyword_search.get_statistics()
                
                stats['domains'][domain] = {
                    'processed_files': len(kb.processed_files),
                    'files': list(kb.processed_files),
                    'vector_count': vector_stats.get('total_documents', 0),
                    'graph_nodes': graph_stats.get('total_nodes', 0),
                    'graph_edges': graph_stats.get('total_edges', 0),
                    'keyword_vocab': keyword_stats.get('vocabulary_size', keyword_stats.get('vocab_size', 0))
                }
            except Exception as e:
                logger.warning(f"Error getting stats for domain {domain}: {e}")
                stats['domains'][domain] = {
                    'processed_files': len(kb.processed_files),
                    'files': list(kb.processed_files),
                    'vector_count': 0,
                    'graph_nodes': 0,
                    'graph_edges': 0,
                    'keyword_vocab': 0
                }
        
        return stats
    
    def list_domains(self) -> List[str]:
        """List all available domains"""
        return list(self.domain_kbs.keys())


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found")
        exit(1)
    
    print("="*60)
    print("Complete Pipeline v2 - Domain-Based KBs")
    print("Intelligent Routing + Incremental Processing")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CompletePipeline2(
        base_dir="data2",
        chunk_size=800,
        chunk_overlap=150,
        use_llm_extraction=False,
        rerank_method="simple"
    )
    
    # Configure PDFs with their domains
    pdf_configs = [
        # Base KB - always included
        {
            'path': 'data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf',
            'domain': 'base'
        },
        # Domain-specific PDFs (auto-classified from filename)
        {
            'path': 'data/pdfs/Children and Vedic Astrology - PDFCOFFEE.COM.pdf'
            # Will auto-classify as 'children'
        },
        {
            'path': 'data/pdfs/ilide.info-jyotish-predicting-marriage-trivedi-pr_a1a475893f2c8116372485aad029d3be.pdf'
            # Will auto-classify as 'marriage'
        },
        {
            'path': 'data/pdfs/Lal kitab - Effects And Remedies.pdf'
            # Will auto-classify as 'remedies'
        }
    ]
    
    # Check which files exist
    existing_configs = [
        config for config in pdf_configs 
        if Path(config['path']).exists()
    ]
    
    if existing_configs:
        print(f"\nFound {len(existing_configs)} PDF file(s) to process")
        
        # Process all PDFs (incremental - skips already processed)
        result = pipeline.process_pdfs(
            pdf_configs=existing_configs,
            force_reprocess=False  # Set True to reprocess all
        )
        
        if result['processed'] > 0 or result['skipped'] > 0:
            # Show statistics
            print("\n" + "="*60)
            print("Domain Knowledge Bases Statistics")
            print("="*60)
            
            stats = pipeline.get_statistics()
            for domain, domain_stats in stats['domains'].items():
                print(f"\n{domain.upper()}:")
                print(f"  Files: {', '.join(domain_stats['files'])}")
                print(f"  Vectors: {domain_stats['vector_count']}")
                print(f"  Graph: {domain_stats['graph_nodes']} nodes, {domain_stats['graph_edges']} edges")
                print(f"  Keywords: {domain_stats['keyword_vocab']} terms")
            
            # Test intelligent routing
            print("\n" + "="*60)
            print("Testing Intelligent Query Routing")
            print("="*60)
            
            test_queries = [
                "What are indicators for children in birth chart?",
                "When will I get married?",
                "What remedies for weak Mars?",
                "Mars in 10th house career effects"
            ]
            
            for query in test_queries:
                print(f"\n{'='*60}")
                print(f"Query: '{query}'")
                print(f"{'='*60}")
                
                # Auto-route and search
                results = pipeline.search(
                    query=query,
                    top_k=3,
                    domains=None,  # Auto-route
                    rerank=True
                )
                
                for i, r in enumerate(results):
                    print(f"\nResult {i+1} [Domain: {r['domain']}]:")
                    print(f"  Score: {r.get('rerank_score', r.get('fusion_score', 0)):.3f}")
                    print(f"  Source: {r['metadata'].get('source', '?')}")
                    print(f"  Page: {r['metadata'].get('page_number', '?')}")
                    print(f"  Content: {r['content'][:120]}...")
        
        else:
            print("\nNo files processed or available")
    
    else:
        print(f"\nNo PDF files found!")
        print("Please add PDFs to the locations specified in pdf_configs")
    
    print("\n" + "="*60)
    print("Pipeline Features:")
    print("  ✓ Domain-based knowledge bases")
    print("  ✓ Intelligent query routing")
    print("  ✓ Incremental processing (skip processed files)")
    print("  ✓ Base KB always included")
    print("  ✓ Auto-classification from filename")
    print("="*60)