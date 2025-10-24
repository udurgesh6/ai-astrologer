"""
PDF Text Extraction Module
Extracts text from PDF files while preserving structure and metadata
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents content from a single page"""
    page_num: int
    text: str
    metadata: Dict


class PDFExtractor:
    """Extracts text from PDF files with metadata preservation"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.doc = None
        self.metadata = {}
        
    def extract_text(self) -> Dict:
        """
        Extract all text from PDF with metadata
        
        Returns:
            Dictionary containing:
                - pages: List of PageContent objects
                - metadata: PDF metadata
                - total_pages: Number of pages
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            self.metadata = self._extract_metadata()
            
            pages = []
            for page_num in range(len(self.doc)):
                page_content = self.extract_page(page_num)
                pages.append(page_content)
            
            logger.info(f"Successfully extracted {len(pages)} pages from {self.pdf_path.name}")
            
            return {
                'pages': pages,
                'metadata': self.metadata,
                'total_pages': len(pages),
                'source': str(self.pdf_path)
            }
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise
        finally:
            if self.doc:
                self.doc.close()
    
    def extract_page(self, page_num: int) -> PageContent:
        """
        Extract text from a specific page
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            PageContent object with text and metadata
        """
        try:
            page = self.doc[page_num]
            text = page.get_text("text")
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Extract page metadata
            page_metadata = {
                'page_number': page_num + 1,  # 1-indexed for user display
                'width': page.rect.width,
                'height': page.rect.height,
            }
            
            return PageContent(
                page_num=page_num + 1,
                text=cleaned_text,
                metadata=page_metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {str(e)}")
            return PageContent(
                page_num=page_num + 1,
                text="",
                metadata={'error': str(e)}
            )
    
    def _extract_metadata(self) -> Dict:
        """Extract PDF metadata"""
        try:
            metadata = self.doc.metadata
            return {
                'title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
            }
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")
            return {}
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (repeated text at start/end of lines)
        # This is a simple approach, might need refinement
        
        # Remove special characters but keep important ones
        # Keep: periods, commas, hyphens, parentheses, apostrophes
        # text = re.sub(r'[^\w\s.,\-()\'\"°]', '', text)
        
        # Fix common OCR issues (if PDF was scanned)
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_text_by_page_range(self, start_page: int, end_page: int) -> str:
        """
        Get text from a range of pages
        
        Args:
            start_page: Starting page (1-indexed)
            end_page: Ending page (1-indexed, inclusive)
            
        Returns:
            Combined text from all pages in range
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            text_parts = []
            
            for page_num in range(start_page - 1, min(end_page, len(self.doc))):
                page_content = self.extract_page(page_num)
                text_parts.append(page_content.text)
            
            return "\n\n".join(text_parts)
            
        finally:
            if self.doc:
                self.doc.close()
    
    @staticmethod
    def validate_pdf(pdf_path: str) -> bool:
        """
        Validate if file is a readable PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            doc = fitz.open(pdf_path)
            is_valid = len(doc) > 0
            doc.close()
            return is_valid
        except Exception as e:
            logger.error(f"Invalid PDF: {str(e)}")
            return False
    
    def get_document_info(self) -> Dict:
        """
        Get basic information about the PDF
        
        Returns:
            Dictionary with document information
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            info = {
                'filename': self.pdf_path.name,
                'total_pages': len(self.doc),
                'file_size_mb': self.pdf_path.stat().st_size / (1024 * 1024),
                'is_encrypted': self.doc.is_encrypted,
                'metadata': self._extract_metadata()
            }
            return info
        finally:
            if self.doc:
                self.doc.close()


# Example usage
if __name__ == "__main__":
    # Test the extractor
    pdf_path = "data/pdfs/sample_astrology.pdf"
    
    if Path(pdf_path).exists():
        extractor = PDFExtractor(pdf_path)
        
        # Get document info
        info = extractor.get_document_info()
        print(f"Document Info: {info}")
        
        # Extract all text
        result = extractor.extract_text()
        print(f"\nExtracted {result['total_pages']} pages")
        print(f"First page text (first 500 chars):\n{result['pages'][0].text[:500]}")
    else:
        print(f"Please place a PDF file at: {pdf_path}")