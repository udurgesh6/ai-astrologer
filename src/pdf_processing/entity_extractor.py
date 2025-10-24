"""
Entity Extraction Module
Extracts astrological entities and relationships from text using regex and LLM
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from entity_dictionary import AstroEntityDictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity"""
    type: str  # planet, house, sign, nakshatra, dasha, aspect, yoga
    value: str  # The actual entity name (normalized)
    original_text: str  # Original text from document
    context: str = ""  # Surrounding context
    position: int = 0  # Position in text
    confidence: float = 1.0  # Confidence score (0-1)
    
@dataclass
class Relationship:
    """Represents a relationship between two entities"""
    entity1: Entity
    entity2: Entity
    relationship_type: str  # placed_in, aspected_by, rules, etc.
    context: str = ""
    confidence: float = 1.0


class RegexEntityExtractor:
    """Extract entities using regex patterns"""
    
    def __init__(self):
        """Initialize with entity dictionary"""
        self.dictionary = AstroEntityDictionary()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.compiled_patterns = {}
        
        # Create patterns for each category
        for category, entities in self.dictionary.get_all_entities().items():
            patterns = []
            for entity_name, variations in entities.items():
                # Create pattern that captures the entity
                variations_pattern = '|'.join(re.escape(v) for v in variations)
                patterns.append((entity_name, re.compile(f'\\b({variations_pattern})\\b', re.IGNORECASE)))
            
            self.compiled_patterns[category] = patterns
    
    def extract_entities(self, text: str, context_window: int = 50) -> List[Entity]:
        """
        Extract all entities from text
        
        Args:
            text: Text to extract from
            context_window: Characters before/after entity for context
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        for category, patterns in self.compiled_patterns.items():
            for entity_name, pattern in patterns:
                for match in pattern.finditer(text):
                    # Get context
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    context = text[start:end]
                    
                    entity = Entity(
                        type=category.rstrip('s'),  # Remove plural 's'
                        value=entity_name,
                        original_text=match.group(),
                        context=context,
                        position=match.start(),
                        confidence=1.0
                    )
                    entities.append(entity)
        
        # Remove duplicates at same position
        entities = self._remove_duplicates(entities)
        
        logger.debug(f"Extracted {len(entities)} entities using regex")
        return entities
    
    def _remove_duplicates(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities at same position"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.type, entity.value, entity.position)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, text: str) -> List[Relationship]:
        """
        Extract relationships between entities
        
        Args:
            text: Text to extract from
            
        Returns:
            List of Relationship objects
        """
        relationships = []
        
        # First extract all entities
        entities = self.extract_entities(text)
        
        if len(entities) < 2:
            return relationships
        
        # Get relationship patterns
        rel_patterns = self.dictionary.get_relationship_patterns()
        
        # Look for relationships between nearby entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Only consider entities close to each other (within 100 chars)
                if abs(entity2.position - entity1.position) > 100:
                    continue
                
                # Get text between entities
                start = min(entity1.position, entity2.position)
                end = max(entity1.position + len(entity1.original_text), 
                         entity2.position + len(entity2.original_text))
                between_text = text[start:end]
                
                # Check for relationship keywords
                for rel_type, keywords in rel_patterns.items():
                    for keyword in keywords:
                        if keyword.lower() in between_text.lower():
                            relationship = Relationship(
                                entity1=entity1,
                                entity2=entity2,
                                relationship_type=rel_type,
                                context=between_text,
                                confidence=0.8
                            )
                            relationships.append(relationship)
                            break
        
        logger.debug(f"Extracted {len(relationships)} relationships using regex")
        return relationships
    
    def extract_compound_entities(self, text: str) -> List[Dict]:
        """
        Extract compound entities like "Mars in 10th house"
        
        Returns:
            List of dictionaries with compound entity info
        """
        compounds = []
        
        # Pattern: Planet in House
        pattern = r'\b(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu|Surya|Chandra|Mangal|Budh|Guru|Shukra|Shani)\s+(?:in|placed in|posited in)\s+(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house\b'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            compounds.append({
                'type': 'planet_in_house',
                'planet': match.group(1),
                'house': match.group(2),
                'full_text': match.group(),
                'position': match.start()
            })
        
        # Pattern: Planet aspects Planet
        pattern = r'\b(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\s+(?:aspects?|aspected by)\s+(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\b'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            compounds.append({
                'type': 'planet_aspect_planet',
                'planet1': match.group(1),
                'planet2': match.group(2),
                'full_text': match.group(),
                'position': match.start()
            })
        
        logger.debug(f"Extracted {len(compounds)} compound entities")
        return compounds


class LLMEntityExtractor:
    """Extract entities using LLM (Claude) for complex cases"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM extractor
        
        Args:
            api_key: Anthropic API key (optional, will use env if not provided)
        """
        try:
            from anthropic import Anthropic
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found. LLM extraction will be disabled.")
                self.client = None
            else:
                self.client = Anthropic(api_key=api_key)
                logger.info("LLM Entity Extractor initialized")
                
        except ImportError:
            logger.warning("Anthropic package not installed. LLM extraction disabled.")
            self.client = None
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using Claude
        
        Args:
            text: Text to extract from
            
        Returns:
            List of Entity objects
        """
        if not self.client:
            logger.warning("LLM client not available")
            return []
        
        prompt = f"""Extract all astrological entities from the following text.

Text: "{text}"

Extract:
1. Planets (Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu)
2. Houses (1st through 12th)
3. Signs (Aries through Pisces)
4. Nakshatras
5. Dashas (planetary periods)
6. Yogas (planetary combinations)
7. Aspects

Return as JSON list:
[
  {{"type": "planet", "value": "Mars", "original_text": "Mars"}},
  {{"type": "house", "value": "10th", "original_text": "10th house"}},
  ...
]

Only return the JSON, no other text."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            import json
            content = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
                
                entities = []
                for item in entities_data:
                    entity = Entity(
                        type=item.get('type', ''),
                        value=item.get('value', ''),
                        original_text=item.get('original_text', ''),
                        confidence=0.9  # LLM extraction confidence
                    )
                    entities.append(entity)
                
                logger.debug(f"LLM extracted {len(entities)} entities")
                return entities
            else:
                logger.warning("Could not parse LLM response")
                return []
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return []
    
    def extract_relationships(self, text: str) -> List[Relationship]:
        """Extract relationships using Claude"""
        if not self.client:
            return []
        
        prompt = f"""Extract relationships between astrological entities in this text.

Text: "{text}"

Find relationships like:
- "Mars in 10th house" → Mars placed_in 10th house
- "Saturn aspects Mars" → Saturn aspected_by Mars  
- "Jupiter rules 9th house" → Jupiter rules 9th house

Return as JSON:
[
  {{
    "entity1": {{"type": "planet", "value": "Mars"}},
    "entity2": {{"type": "house", "value": "10th"}},
    "relationship_type": "placed_in"
  }},
  ...
]

Only return JSON."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            content = response.content[0].text
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                rels_data = json.loads(json_match.group())
                
                relationships = []
                for item in rels_data:
                    e1_data = item.get('entity1', {})
                    e2_data = item.get('entity2', {})
                    
                    entity1 = Entity(
                        type=e1_data.get('type', ''),
                        value=e1_data.get('value', ''),
                        original_text=e1_data.get('value', '')
                    )
                    
                    entity2 = Entity(
                        type=e2_data.get('type', ''),
                        value=e2_data.get('value', ''),
                        original_text=e2_data.get('value', '')
                    )
                    
                    relationship = Relationship(
                        entity1=entity1,
                        entity2=entity2,
                        relationship_type=item.get('relationship_type', ''),
                        confidence=0.9
                    )
                    relationships.append(relationship)
                
                logger.debug(f"LLM extracted {len(relationships)} relationships")
                return relationships
            else:
                return []
                
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {str(e)}")
            return []


class HybridEntityExtractor:
    """Combines regex and LLM extraction for best results"""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize hybrid extractor
        
        Args:
            use_llm: Whether to use LLM extraction
        """
        self.regex_extractor = RegexEntityExtractor()
        self.llm_extractor = LLMEntityExtractor() if use_llm else None
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using both regex and LLM
        
        Args:
            text: Text to extract from
            
        Returns:
            Combined list of entities
        """
        # Always use regex (fast and accurate for clear matches)
        regex_entities = self.regex_extractor.extract_entities(text)
        
        # Use LLM for complex cases if available
        if self.llm_extractor and len(text) < 1000:  # Only for shorter texts
            llm_entities = self.llm_extractor.extract_entities(text)
            
            # Merge results, preferring regex when overlap
            all_entities = self._merge_entities(regex_entities, llm_entities)
            return all_entities
        
        return regex_entities
    
    def extract_relationships(self, text: str) -> List[Relationship]:
        """
        Extract relationships using both methods
        
        Args:
            text: Text to extract from
            
        Returns:
            Combined list of relationships
        """
        # Regex relationships
        regex_rels = self.regex_extractor.extract_relationships(text)
        
        # LLM relationships for complex cases
        if self.llm_extractor and len(text) < 1000:
            llm_rels = self.llm_extractor.extract_relationships(text)
            return regex_rels + llm_rels
        
        return regex_rels
    
    def _merge_entities(self, regex_entities: List[Entity], llm_entities: List[Entity]) -> List[Entity]:
        """
        Merge entities from both sources, avoiding duplicates
        
        Args:
            regex_entities: Entities from regex
            llm_entities: Entities from LLM
            
        Returns:
            Merged list with no duplicates
        """
        # Create set of regex entity values for quick lookup
        regex_values = {(e.type, e.value.lower()) for e in regex_entities}
        
        # Add LLM entities that aren't already found by regex
        merged = list(regex_entities)
        for llm_entity in llm_entities:
            key = (llm_entity.type, llm_entity.value.lower())
            if key not in regex_values:
                merged.append(llm_entity)
        
        return merged
    
    def extract_from_chunk(self, chunk, include_relationships: bool = True) -> Dict:
        """
        Extract entities and relationships from a chunk
        
        Args:
            chunk: Chunk object from TextChunker
            include_relationships: Whether to extract relationships
            
        Returns:
            Dictionary with entities and relationships
        """
        text = chunk.content
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Extract relationships if requested
        relationships = []
        if include_relationships:
            relationships = self.extract_relationships(text)
        
        # Also get compound entities
        compounds = self.regex_extractor.extract_compound_entities(text)
        
        return {
            'chunk_id': chunk.chunk_id,
            'entities': entities,
            'relationships': relationships,
            'compounds': compounds,
            'entity_count': len(entities),
            'relationship_count': len(relationships)
        }
    
    def tag_chunk_with_entities(self, chunk) -> Dict:
        """
        Tag a chunk with extracted entities in metadata
        
        Args:
            chunk: Chunk object
            
        Returns:
            Updated chunk metadata
        """
        extraction = self.extract_from_chunk(chunk, include_relationships=False)
        
        # ChromaDB only accepts str, int, float, bool in metadata
        # So we convert lists to comma-separated strings
        
        entity_values = [e.value for e in extraction['entities']]
        entity_types = list(set(e.type for e in extraction['entities']))
        
        # Store as comma-separated strings
        chunk.metadata['entity_values'] = ', '.join(entity_values) if entity_values else ''
        chunk.metadata['entity_types'] = ', '.join(entity_types) if entity_types else ''
        chunk.metadata['entity_count'] = len(extraction['entities'])
        
        return chunk.metadata


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Entity Extraction")
    print("="*60)
    
    # Sample astrological text
    sample_text = """
    Mars in the 10th house gives strong career ambitions and leadership qualities. 
    When Saturn aspects Mars, it can create delays but also adds discipline.
    
    During Sun Mahadasha, the native experiences recognition and authority. 
    Jupiter in the 2nd house in Sagittarius indicates wealth and prosperity.
    
    The Gaja Kesari Yoga is formed when Jupiter and Moon are in kendras. 
    Venus in 7th house is excellent for marriage and partnerships.
    """
    
    print("\nSample Text:")
    print(sample_text)
    print("\n" + "="*60)
    
    # Test Regex Extractor
    print("\n1. REGEX EXTRACTION")
    print("-"*60)
    
    regex_extractor = RegexEntityExtractor()
    entities = regex_extractor.extract_entities(sample_text)
    
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.type}: {entity.value} ('{entity.original_text}')")
    
    # Test relationships
    relationships = regex_extractor.extract_relationships(sample_text)
    print(f"\nFound {len(relationships)} relationships:")
    for rel in relationships[:5]:  # Show first 5
        print(f"  - {rel.entity1.value} --[{rel.relationship_type}]--> {rel.entity2.value}")
    
    # Test compound entities
    compounds = regex_extractor.extract_compound_entities(sample_text)
    print(f"\nFound {len(compounds)} compound entities:")
    for comp in compounds:
        print(f"  - {comp['type']}: {comp['full_text']}")
    
    # Test Hybrid Extractor (without LLM if key not available)
    print("\n" + "="*60)
    print("2. HYBRID EXTRACTION (Regex + LLM)")
    print("-"*60)
    
    hybrid_extractor = HybridEntityExtractor(use_llm=False)  # Set to True if you have Anthropic key
    
    entities = hybrid_extractor.extract_entities(sample_text)
    print(f"\nFound {len(entities)} total entities")
    
    # Group by type
    by_type = {}
    for entity in entities:
        by_type.setdefault(entity.type, []).append(entity.value)
    
    print("\nEntities by type:")
    for etype, values in by_type.items():
        print(f"  {etype}: {', '.join(set(values))}")
    
    # Test with chunk
    print("\n" + "="*60)
    print("3. CHUNK TAGGING")
    print("-"*60)
    
    from text_chunker import Chunk
    
    test_chunk = Chunk(
        content=sample_text,
        metadata={'page': 1, 'source': 'test.pdf'}
    )
    
    result = hybrid_extractor.extract_from_chunk(test_chunk)
    
    print(f"\nChunk Analysis:")
    print(f"  - Entities found: {result['entity_count']}")
    print(f"  - Relationships found: {result['relationship_count']}")
    print(f"  - Compound entities: {len(result['compounds'])}")
    
    # Tag chunk
    updated_metadata = hybrid_extractor.tag_chunk_with_entities(test_chunk)
    print(f"\nUpdated chunk metadata:")
    print(f"  - Entity types: {updated_metadata['entity_types']}")
    print(f"  - Total entities: {updated_metadata['entity_count']}")
    
    print("\n" + "="*60)
    print("✓ Entity Extraction Tests Complete")
    print("="*60)