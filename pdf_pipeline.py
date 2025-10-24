"""
PDF Processing Pipeline
End-to-end pipeline: PDF → Text Extraction → Chunking → Embeddings → Vector Storage
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import time
from tqdm import tqdm

# Import our modules
from src.pdf_processing.pdf_extractor import PDFExtractor
from src.pdf_processing.text_chunker import TextChunker, Chunk
from src.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Complete pipeline for processing PDFs into vector store"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        chunking_strategy: str = "paragraphs"
    ):
        """
        Initialize pipeline
        
        Args:
            vector_store: VectorStore instance (creates default if None)
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            chunking_strategy: "paragraphs" or "fixed"
        """
        self.vector_store = vector_store or VectorStore()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunking_strategy = chunking_strategy
        
        logger.info("PDF Processing Pipeline initialized")
    
    def process_pdf(
        self,
        pdf_path: str,
        replace_existing: bool = True
    ) -> Dict:
        """
        Process a single PDF through the entire pipeline
        
        Args:
            pdf_path: Path to PDF file
            replace_existing: If True, delete existing docs from this source first
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting to process: {pdf_path.name}")
        
        try:
            # Step 1: Validate PDF
            if not PDFExtractor.validate_pdf(str(pdf_path)):
                raise ValueError(f"Invalid or unreadable PDF: {pdf_path}")
            
            # Step 2: Extract text from PDF
            logger.info("Step 1/4: Extracting text from PDF...")
            extractor = PDFExtractor(str(pdf_path))
            
            # Get document info
            doc_info = extractor.get_document_info()
            logger.info(f"Document info: {doc_info['total_pages']} pages, "
                       f"{doc_info['file_size_mb']:.2f} MB")
            
            # Extract all text
            extraction_result = extractor.extract_text()
            pages = extraction_result['pages']
            
            if not pages:
                raise ValueError("No text extracted from PDF")
            
            logger.info(f"Extracted text from {len(pages)} pages")
            
            # Step 3: Chunk the text
            logger.info("Step 2/4: Chunking text...")
            chunks = self.chunker.chunk_document(
                pages=pages,
                source=pdf_path.name,
                strategy=self.chunking_strategy
            )
            
            if not chunks:
                raise ValueError("No chunks created from text")
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Calculate statistics
            avg_chunk_size = sum(c.token_count for c in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} tokens")
            
            # Step 4: Delete existing documents from this source if requested
            if replace_existing:
                logger.info("Step 3/4: Removing existing documents from this source...")
                self.vector_store.delete_by_source(pdf_path.name)
            
            # Step 5: Add chunks to vector store
            logger.info("Step 4/4: Adding chunks to vector store...")
            logger.info("(This may take a while as embeddings are generated...)")
            
            self.vector_store.add_documents(chunks)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Prepare results
            results = {
                'success': True,
                'source': pdf_path.name,
                'total_pages': len(pages),
                'total_chunks': len(chunks),
                'avg_chunk_size_tokens': avg_chunk_size,
                'processing_time_seconds': processing_time,
                'metadata': extraction_result['metadata']
            }
            
            logger.info(f"\n✓ Successfully processed {pdf_path.name}")
            logger.info(f"  Total chunks: {len(chunks)}")
            logger.info(f"  Processing time: {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                'success': False,
                'source': pdf_path.name,
                'error': str(e)
            }
    
    def process_multiple_pdfs(
        self,
        pdf_paths: list,
        replace_existing: bool = True
    ) -> Dict:
        """
        Process multiple PDFs
        
        Args:
            pdf_paths: List of PDF file paths
            replace_existing: If True, replace existing docs
            
        Returns:
            Summary of all processing results
        """
        logger.info(f"Processing {len(pdf_paths)} PDFs...")
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            result = self.process_pdf(pdf_path, replace_existing)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Calculate summary statistics
        total_chunks = sum(r.get('total_chunks', 0) for r in results if r['success'])
        total_pages = sum(r.get('total_pages', 0) for r in results if r['success'])
        total_time = sum(r.get('processing_time_seconds', 0) for r in results if r['success'])
        
        summary = {
            'total_pdfs': len(pdf_paths),
            'successful': successful,
            'failed': failed,
            'total_pages_processed': total_pages,
            'total_chunks_created': total_chunks,
            'total_processing_time_seconds': total_time,
            'results': results
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Summary:")
        logger.info(f"  Total PDFs: {len(pdf_paths)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"{'='*50}\n")
        
        return summary
    
    def test_retrieval(self, query: str, k: int = 5) -> None:
        """
        Test retrieval with a sample query
        
        Args:
            query: Test query
            k: Number of results to retrieve
        """
        logger.info(f"\nTesting retrieval with query: '{query}'")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results):
            print(f"{'='*60}")
            print(f"Result {i+1}:")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Page: {result['metadata'].get('page_number', 'Unknown')}")
            print(f"Distance: {result.get('distance', 'N/A'):.4f}")
            print(f"\nContent:\n{result['content'][:300]}...")
            print(f"{'='*60}\n")
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline and vector store"""
        stats = self.vector_store.get_collection_stats()
        stats['chunker_config'] = {
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.chunk_overlap,
            'strategy': self.chunking_strategy
        }
        return stats


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please create a .env file with your OpenAI API key")
        exit(1)
    
    # Initialize pipeline
    print("Initializing PDF Processing Pipeline...")
    pipeline = PDFProcessingPipeline(
        chunk_size=800,
        chunk_overlap=150,
        chunking_strategy="paragraphs"
    )
    
    # Example: Process a single PDF
    pdf_path = "data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf"
    
    if Path(pdf_path).exists():
        print(f"\nProcessing: {pdf_path}")
        result = pipeline.process_pdf(pdf_path)
        
        if result['success']:
            print("\n✓ PDF processed successfully!")
            
            # Test retrieval
            print("\n" + "="*60)
            print("Testing Retrieval")
            print("="*60)
            
            test_queries = [
                "What determines career success?",
                "Mars in 10th house effects",
                "Sun Mahadasha results"
            ]
            
            for query in test_queries:
                pipeline.test_retrieval(query, k=3)
                input("\nPress Enter to continue to next query...")
            
            # Get stats
            stats = pipeline.get_pipeline_stats()
            print("\nPipeline Statistics:")
            print(f"Total documents in store: {stats['total_documents']}")
            print(f"Chunk size: {stats['chunker_config']['chunk_size']} tokens")
            print(f"Chunk overlap: {stats['chunker_config']['chunk_overlap']} tokens")
            
        else:
            print(f"\n✗ Failed to process PDF: {result.get('error')}")
    
    else:
        print(f"\nERROR: PDF file not found at: {pdf_path}")
        print("\nTo test the pipeline:")
        print("1. Create directory: data/pdfs/")
        print("2. Place an astrology PDF in that directory")
        print("3. Update the pdf_path variable above")
        print("4. Run this script again")
        
        # Example: Process multiple PDFs
        print("\n" + "="*60)
        print("Example: Processing Multiple PDFs")
        print("="*60)
        
        print("""
# To process multiple PDFs:
pdf_paths = [
    "data/pdfs/astrology_book1.pdf",
    "data/pdfs/astrology_book2.pdf",
    "data/pdfs/astrology_book3.pdf"
]

summary = pipeline.process_multiple_pdfs(pdf_paths)
print(summary)
        """)