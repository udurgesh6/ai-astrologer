"""
Knowledge Graph Module
Builds and manages a graph of astrological entities and their relationships
"""

import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
import logging
import pickle
from pathlib import Path
from entity_extractor import Entity, Relationship

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Graph-based representation of astrological knowledge"""
    
    def __init__(self):
        """Initialize an empty knowledge graph"""
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.entity_index = {}  # Quick lookup: entity_id -> node data
        
        logger.info("Knowledge graph initialized")
    
    def add_entity(self, entity: Entity, chunk_id: Optional[str] = None) -> str:
        """
        Add an entity to the graph
        
        Args:
            entity: Entity object to add
            chunk_id: ID of chunk where entity was found
            
        Returns:
            Node ID in the graph
        """
        # Create unique node ID
        node_id = f"{entity.type}_{entity.value.lower().replace(' ', '_')}"
        
        if not self.graph.has_node(node_id):
            # Add new node
            self.graph.add_node(
                node_id,
                entity_type=entity.type,
                entity_value=entity.value,
                mentions=[],
                chunk_ids=[]
            )
            self.entity_index[node_id] = entity.value
        
        # Update mentions and chunk references
        node_data = self.graph.nodes[node_id]
        if chunk_id and chunk_id not in node_data['chunk_ids']:
            node_data['chunk_ids'].append(chunk_id)
        
        node_data['mentions'].append(entity.original_text)
        
        return node_id
    
    def add_relationship(
        self,
        entity1: Entity,
        entity2: Entity,
        relationship_type: str,
        chunk_id: Optional[str] = None,
        context: str = ""
    ) -> None:
        """
        Add a relationship between two entities
        
        Args:
            entity1: First entity
            entity2: Second entity
            relationship_type: Type of relationship
            chunk_id: Source chunk ID
            context: Context text
        """
        # Add entities if not exists
        node1_id = self.add_entity(entity1, chunk_id)
        node2_id = self.add_entity(entity2, chunk_id)
        
        # Add edge
        self.graph.add_edge(
            node1_id,
            node2_id,
            relationship=relationship_type,
            chunk_id=chunk_id,
            context=context
        )
        
        logger.debug(f"Added relationship: {node1_id} --[{relationship_type}]--> {node2_id}")
    
    def add_from_extraction(self, extraction_result: Dict, chunk_id: str) -> None:
        """
        Add entities and relationships from extraction result
        
        Args:
            extraction_result: Result from HybridEntityExtractor.extract_from_chunk()
            chunk_id: ID of the chunk
        """
        # Add all entities
        for entity in extraction_result['entities']:
            self.add_entity(entity, chunk_id)
        
        # Add all relationships
        for relationship in extraction_result['relationships']:
            self.add_relationship(
                relationship.entity1,
                relationship.entity2,
                relationship.relationship_type,
                chunk_id,
                relationship.context
            )
    
    def get_connected_entities(
        self,
        entity_value: str,
        max_hops: int = 5,
        relationship_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get all entities connected to a given entity
        
        Args:
            entity_value: Entity to start from
            max_hops: Maximum number of hops to traverse
            relationship_filter: Only follow these relationship types
            
        Returns:
            List of connected entities with distance and path
        """
        # Find the node
        node_id = self._find_node_id(entity_value)
        if not node_id:
            logger.warning(f"Entity not found in graph: {entity_value}")
            return []
        
        connected = []
        visited = {node_id}
        
        # BFS traversal
        queue = [(node_id, 0, [node_id])]  # (current_node, distance, path)
        
        while queue:
            current, distance, path = queue.pop(0)
            
            if distance >= max_hops:
                continue
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                # Check relationship filter
                edges = self.graph.get_edge_data(current, neighbor)
                valid_edge = False
                
                for edge_key, edge_data in edges.items():
                    rel_type = edge_data.get('relationship')
                    if relationship_filter is None or rel_type in relationship_filter:
                        valid_edge = True
                        break
                
                if not valid_edge:
                    continue
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                
                # Add to results
                node_data = self.graph.nodes[neighbor]
                connected.append({
                    'node_id': neighbor,
                    'entity_type': node_data['entity_type'],
                    'entity_value': node_data['entity_value'],
                    'distance': distance + 1,
                    'path': new_path,
                    'chunk_ids': node_data['chunk_ids']
                })
                
                # Add to queue for further exploration
                queue.append((neighbor, distance + 1, new_path))
        
        # Sort by distance
        connected.sort(key=lambda x: x['distance'])
        
        logger.debug(f"Found {len(connected)} connected entities for '{entity_value}'")
        return connected
    
    def find_path(self, entity1_value: str, entity2_value: str) -> Optional[List[str]]:
        """
        Find shortest path between two entities
        
        Args:
            entity1_value: First entity
            entity2_value: Second entity
            
        Returns:
            List of node IDs in path, or None if no path
        """
        node1 = self._find_node_id(entity1_value)
        node2 = self._find_node_id(entity2_value)
        
        if not node1 or not node2:
            return None
        
        try:
            path = nx.shortest_path(self.graph, node1, node2)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_entity_info(self, entity_value: str) -> Optional[Dict]:
        """
        Get detailed information about an entity
        
        Args:
            entity_value: Entity to query
            
        Returns:
            Dictionary with entity information
        """
        node_id = self._find_node_id(entity_value)
        if not node_id:
            return None
        
        node_data = self.graph.nodes[node_id]
        
        # Get relationships
        outgoing = []
        for neighbor in self.graph.neighbors(node_id):
            edges = self.graph.get_edge_data(node_id, neighbor)
            for edge_data in edges.values():
                outgoing.append({
                    'target': self.graph.nodes[neighbor]['entity_value'],
                    'relationship': edge_data.get('relationship')
                })
        
        incoming = []
        for predecessor in self.graph.predecessors(node_id):
            edges = self.graph.get_edge_data(predecessor, node_id)
            for edge_data in edges.values():
                incoming.append({
                    'source': self.graph.nodes[predecessor]['entity_value'],
                    'relationship': edge_data.get('relationship')
                })
        
        return {
            'entity_type': node_data['entity_type'],
            'entity_value': node_data['entity_value'],
            'total_mentions': len(node_data['mentions']),
            'chunk_ids': node_data['chunk_ids'],
            'outgoing_relationships': outgoing,
            'incoming_relationships': incoming
        }
    
    def get_chunks_for_entities(self, entity_values: List[str]) -> Set[str]:
        """
        Get all chunk IDs that mention any of the given entities
        
        Args:
            entity_values: List of entity values to search for
            
        Returns:
            Set of chunk IDs
        """
        chunk_ids = set()
        
        for entity_value in entity_values:
            node_id = self._find_node_id(entity_value)
            if node_id:
                node_data = self.graph.nodes[node_id]
                chunk_ids.update(node_data['chunk_ids'])
        
        return chunk_ids
    
    def _find_node_id(self, entity_value: str) -> Optional[str]:
        """Find node ID for an entity value"""
        entity_lower = entity_value.lower().replace(' ', '_')
        
        # Check all possible prefixes
        for node_id in self.graph.nodes():
            if node_id.endswith(entity_lower) or entity_lower in node_id:
                return node_id
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': self._count_node_types(),
            'relationship_types': self._count_relationship_types(),
            'average_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node_id, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type"""
        counts = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def save_graph(self, filepath: str) -> None:
        """
        Save graph to file
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'entity_index': self.entity_index
            }, f)
        
        logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str) -> None:
        """
        Load graph from file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.entity_index = data['entity_index']
        
        logger.info(f"Graph loaded from {filepath}")
    
    def visualize(self, output_file: str = "knowledge_graph.png", max_nodes: int = 50) -> None:
        """
        Create a visualization of the graph
        
        Args:
            output_file: Output file path
            max_nodes: Maximum number of nodes to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get subgraph if too large
            if self.graph.number_of_nodes() > max_nodes:
                # Get most connected nodes
                degrees = dict(self.graph.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                subgraph = self.graph.subgraph(top_nodes)
            else:
                subgraph = self.graph
            
            # Create layout
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # Draw
            plt.figure(figsize=(15, 10))
            
            # Draw nodes by type with different colors
            node_colors = {
                'planet': '#FF6B6B',
                'house': '#4ECDC4',
                'sign': '#45B7D1',
                'yoga': '#FFA07A',
                'dasha': '#98D8C8'
            }
            
            for node_type, color in node_colors.items():
                nodes = [n for n, d in subgraph.nodes(data=True) if d.get('entity_type') == node_type]
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes, 
                                      node_color=color, node_size=500, label=node_type)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True)
            
            # Draw labels
            labels = {n: d['entity_value'] for n, d in subgraph.nodes(data=True)}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
            
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Graph visualization saved to {output_file}")
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Knowledge Graph")
    print("="*60)
    
    # Create graph
    graph = KnowledgeGraph()
    
    # Add some sample entities and relationships
    from entity_extractor import Entity, Relationship
    
    mars = Entity(type='planet', value='Mars', original_text='Mars')
    house_10 = Entity(type='house', value='10th', original_text='10th house')
    saturn = Entity(type='planet', value='Saturn', original_text='Saturn')
    career = Entity(type='signification', value='Career', original_text='career')
    
    # Add entities
    graph.add_entity(mars, chunk_id='chunk_1')
    graph.add_entity(house_10, chunk_id='chunk_1')
    graph.add_entity(saturn, chunk_id='chunk_2')
    graph.add_entity(career, chunk_id='chunk_1')
    
    # Add relationships
    graph.add_relationship(mars, house_10, 'placed_in', 'chunk_1')
    graph.add_relationship(saturn, mars, 'aspected_by', 'chunk_2')
    graph.add_relationship(house_10, career, 'signifies', 'chunk_1')
    
    # Test queries
    print("\n1. Graph Statistics:")
    print("-"*60)
    stats = graph.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n2. Entity Information:")
    print("-"*60)
    mars_info = graph.get_entity_info('Mars')
    if mars_info:
        print(f"  Entity: {mars_info['entity_value']}")
        print(f"  Type: {mars_info['entity_type']}")
        print(f"  Mentions: {mars_info['total_mentions']}")
        print(f"  Outgoing: {mars_info['outgoing_relationships']}")
        print(f"  Incoming: {mars_info['incoming_relationships']}")
    
    print("\n3. Connected Entities:")
    print("-"*60)
    connected = graph.get_connected_entities('Mars', max_hops=5)
    for entity in connected:
        print(f"  - {entity['entity_value']} (distance: {entity['distance']}, path: {' -> '.join(entity['path'])})")
    
    print("\n4. Find Path:")
    print("-"*60)
    path = graph.find_path('Mars', 'Career')
    if path:
        print(f"  Path from Mars to Career: {' -> '.join(path)}")
    
    print("\n" + "="*60)
    print("✓ Knowledge Graph Tests Complete")
    print("="*60)