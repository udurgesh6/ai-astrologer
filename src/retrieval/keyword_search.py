"""
BM25 Keyword Search Module
Traditional keyword-based search using BM25 algorithm
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Tuple
import re
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordSearch:
    """BM25-based keyword search for exact term matching"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize keyword search
        
        Args:
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []  # Store original chunks
        self.chunk_ids = []  # Store chunk IDs
        self.tokenized_corpus = []
        
        logger.info(f"Keyword search initialized (k1={k1}, b={b})")
    
    def index_documents(self, chunks: List) -> None:
        """
        Index document chunks for keyword search
        
        Args:
            chunks: List of Chunk objects from TextChunker
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Indexing {len(chunks)} chunks for keyword search...")
        
        self.chunks = chunks
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Tokenize all chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.content) for chunk in chunks
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"✓ Indexed {len(chunks)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for documents matching query
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of result dictionaries with content and scores
        """
        if not self.bm25:
            logger.warning("BM25 index not built. Call index_documents() first.")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = scores[idx]
            
            # Apply minimum score threshold
            if min_score and score < min_score:
                continue
            
            chunk = self.chunks[idx]
            results.append({
                'id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'score': float(score),
                'rank': len(results) + 1
            })
        
        logger.debug(f"Keyword search found {len(results)} results for '{query}'")
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Keep important punctuation for astrological terms
        # e.g., "10th house", "Mars-Saturn"
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split and remove stopwords
        tokens = text.split()
        
        # Remove very short tokens (< 2 chars) except numbers
        tokens = [
            t for t in tokens 
            if len(t) >= 2 or t.isdigit()
        ]
        
        return tokens
    
    def search_with_highlights(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search and highlight matching terms in results
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Results with highlighted matching terms
        """
        results = self.search(query, top_k)
        query_tokens = set(self._tokenize(query))
        
        for result in results:
            # Find and highlight matching terms
            content = result['content']
            highlights = []
            
            for token in query_tokens:
                # Find all occurrences (case-insensitive)
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    start, end = match.span()
                    # Get context around match (50 chars before/after)
                    context_start = max(0, start - 50)
                    context_end = min(len(content), end + 50)
                    
                    highlight = {
                        'term': match.group(),
                        'context': content[context_start:context_end],
                        'position': start
                    }
                    highlights.append(highlight)
            
            result['highlights'] = highlights
        
        return results
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get number of documents containing a term
        
        Args:
            term: Term to check
            
        Returns:
            Number of documents containing the term
        """
        if not self.bm25:
            return 0
        
        tokens = self._tokenize(term)
        if not tokens:
            return 0
        
        token = tokens[0]
        
        # Count documents containing this token
        count = sum(
            1 for doc_tokens in self.tokenized_corpus 
            if token in doc_tokens
        )
        
        return count
    
    def get_statistics(self) -> Dict:
        """Get keyword search statistics"""
        if not self.bm25:
            return {'indexed': False}
        
        # Get vocabulary size
        vocab = set()
        for doc_tokens in self.tokenized_corpus:
            vocab.update(doc_tokens)
        
        return {
            'indexed': True,
            'total_documents': len(self.chunks),
            'vocabulary_size': len(vocab),
            'avg_doc_length': sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus),
            'k1': self.k1,
            'b': self.b
        }
    
    def save_index(self, filepath: str) -> None:
        """
        Save BM25 index to file
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'chunk_ids': self.chunk_ids,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load BM25 index from file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.chunk_ids = data['chunk_ids']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        
        logger.info(f"BM25 index loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src/pdf_processing')
    from text_chunker import Chunk
    
    print("="*60)
    print("Testing BM25 Keyword Search")
    print("="*60)
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            content="Mars in the 10th house gives strong career ambitions and leadership qualities. "
                   "The native works hard for professional success.",
            metadata={'page': 1, 'source': 'test.pdf'}
        ),
        Chunk(
            content="Sun in 10th house is excellent for career and authority. "
                   "It brings fame, recognition, and success in profession.",
            metadata={'page': 1, 'source': 'test.pdf'}
        ),
        Chunk(
            content="Jupiter in 2nd house indicates wealth and prosperity. "
                   "The person has good fortune with money and finances.",
            metadata={'page': 2, 'source': 'test.pdf'}
        ),
        Chunk(
            content="Venus in 7th house is auspicious for marriage. "
                   "It brings harmony, love, and a beautiful life partner.",
            metadata={'page': 3, 'source': 'test.pdf'}
        ),
        Chunk(
            content="Saturn in 10th house can delay career success but eventually "
                   "brings stability and lasting achievements through hard work.",
            metadata={'page': 1, 'source': 'test.pdf'}
        ),
    ]
    
    # Initialize and index
    keyword_search = KeywordSearch()
    keyword_search.index_documents(sample_chunks)
    
    # Test searches
    test_queries = [
        "career success",
        "10th house",
        "Mars",
        "marriage love",
        "wealth money"
    ]
    
    print("\n" + "="*60)
    print("Test Searches")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-"*60)
        
        results = keyword_search.search(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (score: {result['score']:.2f}):")
            print(f"  Page: {result['metadata']['page']}")
            print(f"  Content: {result['content'][:100]}...")
    
    # Test with highlights
    print("\n" + "="*60)
    print("Search with Highlights")
    print("="*60)
    
    query = "10th house career"
    print(f"\nQuery: '{query}'")
    
    results = keyword_search.search_with_highlights(query, top_k=2)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Highlights:")
        for highlight in result['highlights'][:3]:  # Show first 3
            print(f"    - '{highlight['term']}' in: ...{highlight['context']}...")
    
    # Statistics
    print("\n" + "="*60)
    print("Index Statistics")
    print("="*60)
    
    stats = keyword_search.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Document frequency
    print("\n" + "="*60)
    print("Term Document Frequency")
    print("="*60)
    
    terms = ["career", "10th", "house", "Mars"]
    for term in terms:
        freq = keyword_search.get_document_frequency(term)
        print(f"  '{term}': appears in {freq} documents")
    
    print("\n" + "="*60)
    print("✓ BM25 Keyword Search Tests Complete")
    print("="*60)