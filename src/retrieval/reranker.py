"""
Reranking Module
Reorders search results for better precision using cross-encoder models
"""

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """Rerank search results for better precision"""
    
    def __init__(self, method: str = "simple"):
        """
        Initialize reranker
        
        Args:
            method: Reranking method ("simple", "cohere", or "cross-encoder")
        """
        self.method = method
        self.model = None
        
        if method == "cohere":
            self._init_cohere()
        elif method == "cross-encoder":
            self._init_cross_encoder()
        
        logger.info(f"Reranker initialized with method: {method}")
    
    def _init_cohere(self):
        """Initialize Cohere reranker"""
        try:
            import cohere
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv("COHERE_API_KEY")
            
            if api_key:
                self.model = cohere.Client(api_key)
                logger.info("Cohere reranker initialized")
            else:
                logger.warning("COHERE_API_KEY not found, falling back to simple reranking")
                self.method = "simple"
        except ImportError:
            logger.warning("Cohere package not installed, falling back to simple reranking")
            self.method = "simple"
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            
            # Use a lightweight cross-encoder model
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder model loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to simple reranking")
            self.method = "simple"
        except Exception as e:
            logger.warning(f"Could not load cross-encoder: {str(e)}, falling back to simple")
            self.method = "simple"
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank search results
        
        Args:
            query: Search query
            results: List of result dictionaries
            top_k: Number of results to return (None = all)
            
        Returns:
            Reranked list of results
        """
        if not results:
            return results
        
        if self.method == "cohere" and self.model:
            return self._rerank_cohere(query, results, top_k)
        elif self.method == "cross-encoder" and self.model:
            return self._rerank_cross_encoder(query, results, top_k)
        else:
            return self._rerank_simple(query, results, top_k)
    
    def _rerank_simple(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int]
    ) -> List[Dict]:
        """
        Simple reranking based on:
        1. Number of query terms in content
        2. Position of terms
        3. Existing fusion score
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            content_lower = result['content'].lower()
            
            # Count term matches
            term_matches = sum(1 for term in query_terms if term in content_lower)
            
            # Check if terms appear early in content (first 200 chars)
            early_matches = sum(1 for term in query_terms if term in content_lower[:200])
            
            # Calculate rerank score
            rerank_score = (
                term_matches * 2.0 +  # Term frequency
                early_matches * 1.0 +  # Position bonus
                result.get('fusion_score', 0) * 0.5  # Original score
            )
            
            result['rerank_score'] = rerank_score
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        logger.debug(f"Simple reranking complete: {len(reranked)} results")
        return reranked
    
    def _rerank_cohere(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int]
    ) -> List[Dict]:
        """Rerank using Cohere API"""
        try:
            # Prepare documents
            documents = [result['content'] for result in results]
            
            # Call Cohere rerank
            rerank_results = self.model.rerank(
                query=query,
                documents=documents,
                top_n=top_k or len(documents),
                model='rerank-english-v2.0'
            )
            
            # Map scores back to results
            reranked = []
            for rerank_result in rerank_results:
                idx = rerank_result.index
                result = results[idx].copy()
                result['rerank_score'] = rerank_result.relevance_score
                reranked.append(result)
            
            logger.debug(f"Cohere reranking complete: {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {str(e)}, falling back to simple")
            return self._rerank_simple(query, results, top_k)
    
    def _rerank_cross_encoder(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int]
    ) -> List[Dict]:
        """Rerank using cross-encoder model"""
        try:
            # Prepare query-document pairs
            pairs = [[query, result['content']] for result in results]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            logger.debug(f"Cross-encoder reranking complete: {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}, falling back to simple")
            return self._rerank_simple(query, results, top_k)
    
    def get_relevance_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Get relevance scores for documents without reordering
        
        Args:
            query: Search query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        if self.method == "cross-encoder" and self.model:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        else:
            # Simple scoring
            query_terms = set(query.lower().split())
            scores = []
            for doc in documents:
                doc_lower = doc.lower()
                term_matches = sum(1 for term in query_terms if term in doc_lower)
                scores.append(float(term_matches))
            return scores


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Reranking Module")
    print("="*60)
    
    # Sample results (as if from hybrid retrieval)
    sample_results = [
        {
            'id': 'chunk_1',
            'content': 'Mars in the 10th house gives strong career ambitions and leadership qualities.',
            'metadata': {'page': 1},
            'fusion_score': 0.75
        },
        {
            'id': 'chunk_2',
            'content': 'The 10th house represents career, profession, and public status in astrology.',
            'metadata': {'page': 5},
            'fusion_score': 0.65
        },
        {
            'id': 'chunk_3',
            'content': 'Venus in 7th house is good for marriage and partnerships.',
            'metadata': {'page': 10},
            'fusion_score': 0.50
        },
        {
            'id': 'chunk_4',
            'content': 'Career success is indicated by strong 10th house, Saturn, and Sun placements.',
            'metadata': {'page': 3},
            'fusion_score': 0.60
        },
        {
            'id': 'chunk_5',
            'content': 'Mars gives energy, courage, and determination in professional pursuits.',
            'metadata': {'page': 7},
            'fusion_score': 0.55
        }
    ]
    
    query = "Mars 10th house career"
    
    print(f"\nQuery: '{query}'")
    print(f"Initial results: {len(sample_results)}")
    
    # Test simple reranking
    print("\n" + "="*60)
    print("1. Simple Reranking")
    print("="*60)
    
    reranker_simple = Reranker(method="simple")
    reranked = reranker_simple.rerank(query, sample_results.copy(), top_k=3)
    
    print(f"\nTop 3 after reranking:")
    for i, result in enumerate(reranked):
        print(f"\n{i+1}. Score: {result['rerank_score']:.2f}")
        print(f"   Content: {result['content'][:80]}...")
        print(f"   Original fusion score: {result['fusion_score']:.2f}")
    
    # Test cross-encoder (if available)
    print("\n" + "="*60)
    print("2. Cross-Encoder Reranking")
    print("="*60)
    
    reranker_ce = Reranker(method="cross-encoder")
    
    if reranker_ce.method == "cross-encoder":
        reranked_ce = reranker_ce.rerank(query, sample_results.copy(), top_k=3)
        
        print(f"\nTop 3 after cross-encoder reranking:")
        for i, result in enumerate(reranked_ce):
            print(f"\n{i+1}. Score: {result['rerank_score']:.2f}")
            print(f"   Content: {result['content'][:80]}...")
    else:
        print("Cross-encoder not available (sentence-transformers not installed)")
        print("Install with: pip install sentence-transformers")
    
    # Compare rankings
    print("\n" + "="*60)
    print("3. Ranking Comparison")
    print("="*60)
    
    print("\nOriginal order (by fusion score):")
    original_sorted = sorted(sample_results, key=lambda x: x['fusion_score'], reverse=True)
    for i, r in enumerate(original_sorted[:3]):
        print(f"  {i+1}. {r['content'][:60]}... (score: {r['fusion_score']:.2f})")
    
    print("\nAfter simple reranking:")
    for i, r in enumerate(reranked[:3]):
        print(f"  {i+1}. {r['content'][:60]}... (score: {r['rerank_score']:.2f})")
    
    # Get relevance scores
    print("\n" + "="*60)
    print("4. Relevance Scores")
    print("="*60)
    
    documents = [r['content'] for r in sample_results]
    scores = reranker_simple.get_relevance_scores(query, documents)
    
    print(f"\nRelevance scores for query '{query}':")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        print(f"  Doc {i+1}: {score:.2f} - {doc[:50]}...")
    
    print("\n" + "="*60)
    print("✓ Reranking Module Tests Complete")
    print("="*60)
    
    print("\nReranking improves precision by:")
    print("  - Considering query term frequency")
    print("  - Giving bonus to terms appearing early")
    print("  - Using advanced models (cross-encoder) when available")
    print("  - Combining with original retrieval scores")