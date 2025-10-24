"""
Graph-Based Retrieval Module
Retrieves chunks using knowledge graph relationships
"""

from typing import List, Dict, Set, Optional
import logging
from knowledge_graph import KnowledgeGraph
from entity_extractor import HybridEntityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRetrieval:
    """Retrieve relevant chunks using knowledge graph traversal"""
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        entity_extractor: Optional[HybridEntityExtractor] = None
    ):
        """
        Initialize graph retrieval
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            entity_extractor: Entity extractor for query processing
        """
        self.graph = knowledge_graph
        self.extractor = entity_extractor or HybridEntityExtractor(use_llm=False)
        
        logger.info("Graph retrieval initialized")
    
    def retrieve(
        self,
        query: str,
        max_hops: int = 5,
        top_k: int = 10
    ) -> List[str]:
        """
        Retrieve chunk IDs relevant to query using graph traversal
        
        Args:
            query: Search query
            max_hops: Maximum hops in graph traversal
            top_k: Maximum number of chunk IDs to return
            
        Returns:
            List of chunk IDs
        """
        # Extract entities from query
        query_entities = self.extractor.extract_entities(query)
        
        if not query_entities:
            logger.warning(f"No entities found in query: {query}")
            return []
        
        logger.debug(f"Found {len(query_entities)} entities in query")
        
        # Get all connected entities
        all_connected = set()
        entity_values = [e.value for e in query_entities]
        
        for entity in query_entities:
            connected = self.graph.get_connected_entities(
                entity.value,
                max_hops=max_hops
            )
            
            # Add connected entity values
            for conn in connected:
                all_connected.add(conn['entity_value'])
        
        # Add original query entities
        all_connected.update(entity_values)
        
        logger.debug(f"Found {len(all_connected)} total entities (including connected)")
        
        # Get chunks for all these entities
        chunk_ids = self.graph.get_chunks_for_entities(list(all_connected))
        
        # Rank chunks by relevance (number of query entities they contain)
        ranked_chunks = self._rank_chunks(chunk_ids, entity_values)
        
        # Return top k
        top_chunks = [chunk_id for chunk_id, score in ranked_chunks[:top_k]]
        
        logger.info(f"Graph retrieval found {len(top_chunks)} chunks for query")
        return top_chunks
    
    def _rank_chunks(
        self,
        chunk_ids: Set[str],
        query_entities: List[str]
    ) -> List[tuple]:
        """
        Rank chunks by number of query entities they contain
        
        Args:
            chunk_ids: Set of chunk IDs
            query_entities: List of entity values from query
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score
        """
        chunk_scores = {}
        
        for chunk_id in chunk_ids:
            score = 0
            
            # Count how many query entities are in this chunk
            for entity_value in query_entities:
                entity_chunks = self.graph.get_chunks_for_entities([entity_value])
                if chunk_id in entity_chunks:
                    score += 1
            
            chunk_scores[chunk_id] = score
        
        # Sort by score (descending)
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def get_related_concepts(
        self,
        query: str,
        max_concepts: int = 10
    ) -> List[Dict]:
        """
        Get related astrological concepts for a query
        
        Args:
            query: Search query
            max_concepts: Maximum number of concepts to return
            
        Returns:
            List of related concept dictionaries
        """
        # Extract entities from query
        query_entities = self.extractor.extract_entities(query)
        
        if not query_entities:
            return []
        
        related_concepts = []
        seen = set()
        
        for entity in query_entities:
            connected = self.graph.get_connected_entities(
                entity.value,
                max_hops=2
            )
            
            for conn in connected:
                if conn['entity_value'] not in seen:
                    related_concepts.append({
                        'entity': conn['entity_value'],
                        'type': conn['entity_type'],
                        'distance': conn['distance'],
                        'from': entity.value
                    })
                    seen.add(conn['entity_value'])
        
        # Sort by distance
        related_concepts.sort(key=lambda x: x['distance'])
        
        return related_concepts[:max_concepts]
    
    def explain_connection(
        self,
        entity1: str,
        entity2: str
    ) -> Optional[Dict]:
        """
        Explain how two entities are connected in the graph
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Dictionary with connection information
        """
        path = self.graph.find_path(entity1, entity2)
        
        if not path:
            return None
        
        # Get relationship information for each hop
        connections = []
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            
            # Get edge data
            edges = self.graph.graph.get_edge_data(node1, node2)
            if edges:
                for edge_data in edges.values():
                    connections.append({
                        'from': self.graph.graph.nodes[node1]['entity_value'],
                        'to': self.graph.graph.nodes[node2]['entity_value'],
                        'relationship': edge_data.get('relationship', 'unknown')
                    })
                    break
        
        return {
            'entity1': entity1,
            'entity2': entity2,
            'distance': len(path) - 1,
            'path': [self.graph.graph.nodes[n]['entity_value'] for n in path],
            'connections': connections
        }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Graph-Based Retrieval")
    print("="*60)
    
    # Create sample graph and retrieval
    from knowledge_graph import KnowledgeGraph
    from entity_extractor import Entity, HybridEntityExtractor
    
    # Build sample graph
    graph = KnowledgeGraph()
    
    # Add sample data
    entities = [
        (Entity(type='planet', value='Mars', original_text='Mars'), 'chunk_1'),
        (Entity(type='house', value='10th', original_text='10th house'), 'chunk_1'),
        (Entity(type='signification', value='Career', original_text='career'), 'chunk_1'),
        (Entity(type='planet', value='Saturn', original_text='Saturn'), 'chunk_2'),
        (Entity(type='aspect', value='Aspect', original_text='aspects'), 'chunk_2'),
        (Entity(type='planet', value='Sun', original_text='Sun'), 'chunk_3'),
        (Entity(type='dasha', value='Sun Dasha', original_text='Sun Mahadasha'), 'chunk_3'),
    ]
    
    for entity, chunk_id in entities:
        graph.add_entity(entity, chunk_id)
    
    # Add relationships
    mars = Entity(type='planet', value='Mars', original_text='Mars')
    house_10 = Entity(type='house', value='10th', original_text='10th house')
    career = Entity(type='signification', value='Career', original_text='career')
    saturn = Entity(type='planet', value='Saturn', original_text='Saturn')
    sun = Entity(type='planet', value='Sun', original_text='Sun')
    sun_dasha = Entity(type='dasha', value='Sun Dasha', original_text='Sun Dasha')
    
    graph.add_relationship(mars, house_10, 'placed_in', 'chunk_1')
    graph.add_relationship(house_10, career, 'signifies', 'chunk_1')
    graph.add_relationship(saturn, mars, 'aspects', 'chunk_2')
    graph.add_relationship(sun, house_10, 'placed_in', 'chunk_3')
    graph.add_relationship(sun_dasha, sun, 'relates_to', 'chunk_3')
    
    # Initialize retrieval
    retrieval = GraphRetrieval(graph)
    
    # Test 1: Basic retrieval
    print("\n1. Basic Retrieval:")
    print("-"*60)
    query = "What about Mars and career?"
    chunk_ids = retrieval.retrieve(query, max_hops=5, top_k=5)
    print(f"Query: {query}")
    print(f"Retrieved chunks: {chunk_ids}")
    
    # Test 2: Related concepts
    print("\n2. Related Concepts:")
    print("-"*60)
    related = retrieval.get_related_concepts(query, max_concepts=5)
    print(f"Query: {query}")
    print(f"Related concepts:")
    for concept in related:
        print(f"  - {concept['entity']} ({concept['type']}) - distance: {concept['distance']}")
    
    # Test 3: Explain connection
    print("\n3. Explain Connection:")
    print("-"*60)
    connection = retrieval.explain_connection('Mars', 'Career')
    if connection:
        print(f"Connection from {connection['entity1']} to {connection['entity2']}:")
        print(f"  Distance: {connection['distance']} hops")
        print(f"  Path: {' -> '.join(connection['path'])}")
        print(f"  Relationships:")
        for conn in connection['connections']:
            print(f"    {conn['from']} --[{conn['relationship']}]--> {conn['to']}")
    else:
        print("No connection found")
    
    print("\n" + "="*60)
    print("✓ Graph Retrieval Tests Complete")
    print("="*60)