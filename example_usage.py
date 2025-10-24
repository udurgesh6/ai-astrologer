"""
Example Usage Script
Demonstrates how to use the PDF processing pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')

from pdf_extractor import PDFExtractor
from text_chunker import TextChunker
from vector_store import VectorStore

# Load environment variables
load_dotenv()

def example_1_basic_extraction():
    """Example 1: Basic PDF text extraction"""
    print("\n" + "="*60)
    print("Example 1: Basic PDF Text Extraction")
    print("="*60)
    
    pdf_path = "data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf"
    
    if not Path(pdf_path).exists():
        print(f"⚠ PDF not found: {pdf_path}")
        print("Please add a PDF file to data/pdfs/ directory")
        return
    
    # Extract text
    extractor = PDFExtractor(pdf_path)
    result = extractor.extract_text()
    
    print(f"\n✓ Extracted {result['total_pages']} pages")
    print(f"Document: {result['metadata'].get('title', 'Unknown')}")
    
    # Show first page preview
    if result['pages']:
        first_page = result['pages'][0]
        print(f"\nFirst page preview (first 300 chars):")
        print(f"{first_page.text[:300]}...")

def example_2_chunking():
    """Example 2: Text chunking"""
    print("\n" + "="*60)
    print("Example 2: Text Chunking")
    print("="*60)
    
    sample_text = """
    Mars in the 10th house is a powerful placement for career. It gives strong ambitions, 
    leadership qualities, and the drive to succeed. The native becomes very determined 
    and works hard to achieve professional goals.
    
    However, Mars can also create conflicts with authority figures. There may be 
    aggression or impatience in the workplace. The person should learn to control 
    their temper and channel Mars energy constructively.
    
    When Mars is well-aspected, it can lead to success in fields like engineering, 
    military, sports, or any career requiring courage and energy. The native may 
    become a pioneer in their field.
    """
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_by_paragraphs(sample_text, {"page": 1})
    
    print(f"\n✓ Created {len(chunks)} chunks from sample text")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk.token_count} tokens):")
        print(f"{chunk.content}")
        print(f"Metadata: {chunk.metadata}")

def example_3_vector_search():
    """Example 3: Vector storage and search"""
    print("\n" + "="*60)
    print("Example 3: Vector Storage and Search")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not found. Skipping this example.")
        return
    
    # Create vector store
    vector_store = VectorStore(
        collection_name="example_collection",
        persist_directory="./data/test_chroma"
    )
    
    # Sample astrological texts
    from text_chunker import Chunk
    
    sample_chunks = [
        Chunk(
            content="Mars in 10th house gives strong career ambitions and leadership. "
                   "The person works hard for professional success and recognition.",
            metadata={"source": "example.pdf", "page": 1, "topic": "career"}
        ),
        Chunk(
            content="Sun in 10th house is excellent for career and authority. "
                   "It brings fame, recognition, and success in profession.",
            metadata={"source": "example.pdf", "page": 1, "topic": "career"}
        ),
        Chunk(
            content="Jupiter in 2nd house indicates wealth and prosperity. "
                   "The person has good fortune with money and finances.",
            metadata={"source": "example.pdf", "page": 2, "topic": "wealth"}
        ),
        Chunk(
            content="Venus in 7th house is very auspicious for marriage. "
                   "It brings harmony, love, and a beautiful life partner.",
            metadata={"source": "example.pdf", "page": 3, "topic": "marriage"}
        ),
        Chunk(
            content="Saturn in 10th house can delay career success but eventually "
                   "brings stability and long-lasting achievements through hard work.",
            metadata={"source": "example.pdf", "page": 1, "topic": "career"}
        ),
    ]
    
    print("\nAdding sample chunks to vector store...")
    vector_store.add_documents(sample_chunks)
    
    # Test searches
    test_queries = [
        "career success and professional growth",
        "wealth and money matters",
        "marriage and relationships"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = vector_store.similarity_search(query, k=2)
        
        for i, result in enumerate(results):
            similarity = 1 - result['distance']
            print(f"\nResult {i+1} (similarity: {similarity:.3f}):")
            print(f"Content: {result['content']}")
            print(f"Topic: {result['metadata']['topic']}")
    
    # Cleanup
    print("\n\nCleaning up test collection...")
    vector_store.delete_collection()
    print("✓ Done")

def example_4_complete_pipeline():
    """Example 4: Complete pipeline"""
    print("\n" + "="*60)
    print("Example 4: Complete PDF Processing Pipeline")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not found. Skipping this example.")
        return
    
    from pdf_pipeline import PDFProcessingPipeline
    
    # Initialize pipeline
    pipeline = PDFProcessingPipeline(
        chunk_size=500,
        chunk_overlap=100,
        chunking_strategy="paragraphs"
    )
    
    # Check for PDF
    pdf_path = "data/pdfs/jyotish_elements-of-vedic-astrology_k-s-charak-1.pdf"
    if not Path(pdf_path).exists():
        print(f"⚠ PDF not found: {pdf_path}")
        print("Add a PDF to data/pdfs/ to test the complete pipeline")
        return
    
    # Process PDF
    print(f"\nProcessing: {pdf_path}")
    print("This may take a few minutes...\n")
    
    result = pipeline.process_pdf(pdf_path)
    
    if result['success']:
        print(f"\n✓ Successfully processed!")
        print(f"  Pages: {result['total_pages']}")
        print(f"  Chunks: {result['total_chunks']}")
        print(f"  Time: {result['processing_time_seconds']:.2f}s")
        
        # Test retrieval
        print("\n" + "="*60)
        print("Testing Retrieval")
        print("="*60)
        
        test_query = "What about career and professional success?"
        print(f"\nQuery: {test_query}")
        
        results = pipeline.vector_store.similarity_search(test_query, k=3)
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Page: {result['metadata'].get('page_number', 'N/A')}")
            print(f"Content: {result['content'][:200]}...")
    else:
        print(f"✗ Failed: {result.get('error')}")

def main():
    """Run all examples"""
    print("="*60)
    print("Astrology PDF Q&A - Usage Examples")
    print("="*60)
    
    examples = [
        ("Basic PDF Extraction", example_1_basic_extraction),
        ("Text Chunking", example_2_chunking),
        ("Vector Search", example_3_vector_search),
        ("Complete Pipeline", example_4_complete_pipeline),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
            
            # Wait for user before next example
            print("\n" + "-"*60)
            response = input("Press Enter to continue to next example (or 'q' to quit)...")
            if response.lower() == 'q':
                break
                
        except Exception as e:
            print(f"\n✗ Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the code in each example")
    print("2. Try modifying parameters")
    print("3. Add your own PDFs and test")
    print("4. Move on to Sprint 2: Entity Extraction")

if __name__ == "__main__":
    main()