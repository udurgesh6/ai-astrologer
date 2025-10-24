"""
Text Chunking Module
Splits text into semantic chunks with overlap for optimal retrieval
"""

import re
import tiktoken
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0
    
    def __post_init__(self):
        if not self.token_count and self.content:
            self.token_count = self._count_tokens(self.content)
    
    @staticmethod
    def _count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text.split()) * 1.3


class TextChunker:
    """Splits text into chunks with various strategies"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size in tokens
            chunk_overlap: Number of overlapping tokens
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_by_paragraphs(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text by paragraphs, respecting semantic boundaries
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph
                para_chunks = self._split_large_text(para, metadata)
                chunks.extend(para_chunks)
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = self._count_tokens("\n\n".join(current_chunk))
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if self._count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_fixed_size(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text into fixed-size chunks with overlap
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Clean up chunk boundaries (try to end at sentence)
            chunk_text = self._clean_chunk_boundary(chunk_text)
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata))
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _split_large_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Split text that's too large into smaller chunks"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata))
                
                # Add overlap
                overlap = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap + [sentence]
                current_tokens = self._count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata))
        
        return chunks
    
    def _get_overlap_text(self, paragraphs: List[str]) -> str:
        """Get overlap text from paragraphs"""
        if not paragraphs:
            return ""
        
        # Take last paragraph(s) that fit in overlap size
        overlap_tokens = 0
        overlap_paras = []
        
        for para in reversed(paragraphs):
            para_tokens = self._count_tokens(para)
            if overlap_tokens + para_tokens <= self.chunk_overlap:
                overlap_paras.insert(0, para)
                overlap_tokens += para_tokens
            else:
                break
        
        return "\n\n".join(overlap_paras)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences"""
        overlap_tokens = 0
        overlap = []
        
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap
    
    def _clean_chunk_boundary(self, text: str) -> str:
        """Clean chunk boundaries to end at sentence if possible"""
        # Try to end at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            # If we have multiple sentences, remove partial last sentence
            return " ".join(sentences[:-1])
        return text
    
    def _create_chunk(self, text: str, base_metadata: Optional[Dict] = None) -> Chunk:
        """Create a Chunk object with metadata"""
        metadata = base_metadata.copy() if base_metadata else {}
        
        return Chunk(
            content=text.strip(),
            metadata=metadata,
            token_count=self._count_tokens(text)
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)
    
    def chunk_document(
        self,
        pages: List,
        source: str,
        strategy: str = "paragraphs"
    ) -> List[Chunk]:
        """
        Chunk an entire document
        
        Args:
            pages: List of PageContent objects from PDFExtractor
            source: Source document name
            strategy: "paragraphs" or "fixed"
            
        Returns:
            List of all chunks from document
        """
        all_chunks = []
        
        for page in pages:
            page_metadata = {
                'source': source,
                'page_number': page.page_num,
                **page.metadata
            }
            
            if strategy == "paragraphs":
                chunks = self.chunk_by_paragraphs(page.text, page_metadata)
            else:
                chunks = self.chunk_fixed_size(page.text, page_metadata)
            
            all_chunks.extend(chunks)
        
        # Add sequential chunk numbers
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_index'] = idx
            chunk.metadata['total_chunks'] = len(all_chunks)
        
        logger.info(f"Document chunked into {len(all_chunks)} total chunks")
        return all_chunks


# Example usage
if __name__ == "__main__":
    # Test chunker
    sample_text = """
    Astrology is an ancient science that studies the influence of celestial bodies on human life.
    
    The birth chart, also known as the horoscope or kundli, is a map of the sky at the exact moment of birth. It shows the positions of planets, signs, and houses.
    
    There are 12 houses in a birth chart, each representing different areas of life. The 1st house represents self and personality. The 10th house represents career and public image.
    
    Planets have different significations. Sun represents soul, authority, and father. Moon represents mind, emotions, and mother. Mars represents energy, courage, and siblings.
    
    When analyzing a chart, astrologers look at planet placements, house lordships, aspects, and dashas. All these factors combine to give a complete picture.
    """
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    chunks = chunker.chunk_by_paragraphs(
        sample_text,
        metadata={'source': 'test_doc', 'page_number': 1}
    )
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({chunk.token_count} tokens):")
        print(f"{chunk.content[:200]}...")
        print(f"Metadata: {chunk.metadata}\n")