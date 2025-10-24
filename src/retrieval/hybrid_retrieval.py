"""
Complete Hybrid Retrieval System
Combines Vector Search + Keyword Search + Graph Retrieval
"""

from typing import List, Dict, Set, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetrieval:
    """
    Combines multiple retrieval strategies for best results:
    - Vector Search (semantic similarity)
    - Keyword Search (BM25, exact matching)
    - Graph Retrieval (relationship-based)
    """
    
    def __init__(
        self,
        vector_store=None,
        keyword_search=None,
        graph_retrieval=None
    ):
        """
        Initialize hybrid retrieval
        
        Args:
            vector_store: VectorStore instance
            keyword_search: KeywordSearch instance
            graph_retrieval: GraphRetrieval instance
        """
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.graph_retrieval = graph_retrieval
        
        logger.info("Hybrid retrieval system initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_vector: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ) -> List[Dict]:
        """
        Retrieve using multiple strategies and fuse results
        
        Args:
            query: Search query
            top_k: Number of final results
            use_vector: Use vector similarity search
            use_keyword: Use keyword search
            use_graph: Use graph-based retrieval
            fusion_method: How to combine results ("rrf" or "weighted")
            
        Returns:
            List of ranked results
        """
        all_results = {}  # chunk_id -> result dict
        method_ranks = defaultdict(dict)  # method -> {chunk_id -> rank}
        
        # 1. Vector Search
        if use_vector and self.vector_store:
            logger.debug("Running vector search...")
            vector_results = self.vector_store.similarity_search(query, k=top_k * 2)
            
            for rank, result in enumerate(vector_results):
                chunk_id = result['id']
                
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        'id': chunk_id,
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'sources': []
                    }
                
                all_results[chunk_id]['sources'].append('vector')
                all_results[chunk_id]['vector_score'] = 1 - result.get('distance', 0)
                method_ranks['vector'][chunk_id] = rank + 1
        
        # 2. Keyword Search
        if use_keyword and self.keyword_search:
            logger.debug("Running keyword search...")
            keyword_results = self.keyword_search.search(query, top_k=top_k * 2)
            
            for rank, result in enumerate(keyword_results):
                chunk_id = result['id']
                
                if chunk_id not in all_results:
                    # Fetch content from vector store if not already present
                    vector_result = self.vector_store.collection.get(ids=[chunk_id])
                    if vector_result['documents']:
                        all_results[chunk_id] = {
                            'id': chunk_id,
                            'content': vector_result['documents'][0],
                            'metadata': vector_result['metadatas'][0] if vector_result['metadatas'] else {},
                            'sources': []
                        }
                
                if chunk_id in all_results:
                    all_results[chunk_id]['sources'].append('keyword')
                    all_results[chunk_id]['keyword_score'] = result['score']
                    method_ranks['keyword'][chunk_id] = rank + 1
        
        # 3. Graph Retrieval
        if use_graph and self.graph_retrieval:
            logger.debug("Running graph retrieval...")
            graph_chunk_ids = self.graph_retrieval.retrieve(query, max_hops=2, top_k=top_k * 2)
            
            for rank, chunk_id in enumerate(graph_chunk_ids):
                if chunk_id not in all_results:
                    # Fetch content from vector store
                    vector_result = self.vector_store.collection.get(ids=[chunk_id])
                    if vector_result['documents']:
                        all_results[chunk_id] = {
                            'id': chunk_id,
                            'content': vector_result['documents'][0],
                            'metadata': vector_result['metadatas'][0] if vector_result['metadatas'] else {},
                            'sources': []
                        }
                
                if chunk_id in all_results:
                    all_results[chunk_id]['sources'].append('graph')
                    method_ranks['graph'][chunk_id] = rank + 1
        
        # Fuse results
        if fusion_method == "rrf":
            ranked_results = self._reciprocal_rank_fusion(all_results, method_ranks, k=60)
        else:
            ranked_results = self._weighted_fusion(all_results, method_ranks)
        
        # Return top k
        final_results = ranked_results[:top_k]
        
        logger.info(f"Hybrid retrieval found {len(final_results)} results using {len(method_ranks)} methods")
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict,
        method_ranks: Dict,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        Score = sum(1 / (k + rank)) across all methods
        
        Args:
            all_results: All retrieved chunks
            method_ranks: Rankings from each method
            k: RRF constant (default 60)
            
        Returns:
            Sorted list of results
        """
        rrf_scores = {}
        
        for chunk_id in all_results:
            score = 0.0
            
            # Add RRF score from each method that found this chunk
            for method, ranks in method_ranks.items():
                if chunk_id in ranks:
                    rank = ranks[chunk_id]
                    score += 1.0 / (k + rank)
            
            rrf_scores[chunk_id] = score
            all_results[chunk_id]['fusion_score'] = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [all_results[chunk_id] for chunk_id in sorted_ids]
    
    def _weighted_fusion(
        self,
        all_results: Dict,
        method_ranks: Dict,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Weighted score fusion
        
        Args:
            all_results: All retrieved chunks
            method_ranks: Rankings from each method
            weights: Weight for each method (default: equal weights)
            
        Returns:
            Sorted list of results
        """
        if weights is None:
            weights = {'vector': 0.4, 'keyword': 0.3, 'graph': 0.3}
        
        weighted_scores = {}
        
        for chunk_id in all_results:
            score = 0.0
            
            # Normalize ranks to scores (1/rank)
            for method, ranks in method_ranks.items():
                if chunk_id in ranks:
                    rank = ranks[chunk_id]
                    method_score = 1.0 / rank
                    score += weights.get(method, 0.0) * method_score
            
            weighted_scores[chunk_id] = score
            all_results[chunk_id]['fusion_score'] = score
        
        # Sort by weighted score
        sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
        
        return [all_results[chunk_id] for chunk_id in sorted_ids]
    
    def get_statistics(self) -> Dict:
        """Get statistics about available retrieval methods"""
        stats = {
            'vector_available': self.vector_store is not None,
            'keyword_available': self.keyword_search is not None,
            'graph_available': self.graph_retrieval is not None
        }
        
        if self.vector_store:
            stats['vector_docs'] = self.vector_store.get_collection_stats()['total_documents']
        
        if self.keyword_search:
            keyword_stats = self.keyword_search.get_statistics()
            stats['keyword_indexed'] = keyword_stats.get('indexed', False)
            stats['keyword_vocab_size'] = keyword_stats.get('vocabulary_size', 0)
        
        if self.graph_retrieval:
            graph_stats = self.graph_retrieval.graph.get_statistics()
            stats['graph_nodes'] = graph_stats['total_nodes']
            stats['graph_edges'] = graph_stats['total_edges']
        
        return stats


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Hybrid Retrieval System")
    print("="*60)
    
    print("\nThis is a demonstration of the hybrid retrieval architecture.")
    print("For full testing, run the complete_pipeline.py after Sprint 3 setup.")
    
    print("\nHybrid retrieval combines:")
    print("  1. Vector Search - semantic similarity")
    print("  2. Keyword Search - exact term matching (BM25)")
    print("  3. Graph Retrieval - relationship-based connections")
    
    print("\nFusion methods:")
    print("  - RRF (Reciprocal Rank Fusion): Combines rankings fairly")
    print("  - Weighted: Assigns different importance to each method")
    
    print("\n" + "="*60)
    print("Example Usage:")
    print("="*60)
    
    example_code = '''
from hybrid_retrieval import HybridRetrieval

# Initialize with all three components
hybrid = HybridRetrieval(
    vector_store=vector_store,
    keyword_search=keyword_search,
    graph_retrieval=graph_retrieval
)

# Search using all methods
results = hybrid.retrieve(
    query="Mars in 10th house career",
    top_k=10,
    use_vector=True,
    use_keyword=True,
    use_graph=True,
    fusion_method="rrf"  # Reciprocal Rank Fusion
)

# Results include:
# - Content from all three sources
# - Fusion score (combined ranking)
# - Sources used (which methods found this chunk)
    '''
    
    print(example_code)
    
    print("\n" + "="*60)
    print("✓ Hybrid Retrieval Module Ready")
    print("="*60)