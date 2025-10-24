"""
Orchestration Agent Module
Intelligent query processing and multi-step reasoning
"""

from typing import List, Dict, Optional, Tuple
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """
    Intelligent agent that:
    1. Understands and classifies queries
    2. Decomposes complex questions
    3. Orchestrates retrieval strategies
    4. Synthesizes comprehensive answers
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize orchestration agent
        
        Args:
            api_key: OpenAI API key (optional, uses env if not provided)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Add it to .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4-turbo-preview"  # or "gpt-4" or "gpt-3.5-turbo"
        
        logger.info("Orchestration agent initialized with OpenAI")
    
    def classify_query(self, query: str) -> Dict:
        """
        Classify the query type
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query classification
        """
        prompt = f"""Classify this astrology query into one of these types:

Query: "{query}"

Types:
1. GENERAL_KNOWLEDGE - Asking what factors to check (e.g., "What determines career success?")
2. SPECIFIC_PLACEMENT - Asking about specific placements (e.g., "I have Mars in 10th house, what about my career?")
3. COMPARISON - Comparing two things (e.g., "What's better for career: Sun or Mars in 10th?")
4. TIMING - Asking about timing/dasha (e.g., "What happens in Sun Mahadasha?")
5. GENERAL_QUESTION - General astrology question

Return ONLY one word: the type name."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            
            query_type = response.choices[0].message.content.strip().upper()
            
            # Validate
            valid_types = ["GENERAL_KNOWLEDGE", "SPECIFIC_PLACEMENT", "COMPARISON", "TIMING", "GENERAL_QUESTION"]
            if query_type not in valid_types:
                query_type = "GENERAL_QUESTION"
            
            logger.info(f"Query classified as: {query_type}")
            
            return {
                'type': query_type,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {'type': 'GENERAL_QUESTION', 'query': query}
    
    def decompose_query(self, query: str, query_type: str) -> List[str]:
        """
        Decompose complex query into sub-queries
        
        Args:
            query: User query
            query_type: Query type from classification
            
        Returns:
            List of sub-queries
        """
        if query_type == "GENERAL_KNOWLEDGE":
            # Decompose into factors to check
            prompt = f"""Break down this astrology question into specific factors to check:

Question: "{query}"

List the specific astrological factors that should be analyzed. Return as a simple numbered list.

Example:
Question: "What determines career success?"
1. 10th house lord and placements
2. Saturn position and strength
3. Sun placement
4. Current Mahadasha
5. Jupiter aspects

Now for the given question:"""

        elif query_type == "SPECIFIC_PLACEMENT":
            # Decompose into aspects to analyze
            prompt = f"""Break down this specific placement question into aspects to analyze:

Question: "{query}"

List what should be analyzed. Return as a simple numbered list.

Example:
Question: "I have Mars in 10th house, what about career?"
1. Mars in 10th house general effects
2. Mars dignity (sign position)
3. Aspects to Mars
4. 10th house significations
5. Current dasha influence

Now for the given question:"""

        else:
            # Simple decomposition
            return [query]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse numbered list
            sub_queries = []
            for line in text.split('\n'):
                line = line.strip()
                # Remove numbering
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove number and dot/dash
                    clean = line.lstrip('0123456789.-) ').strip()
                    if clean:
                        sub_queries.append(clean)
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            return [query]
    
    def synthesize_answer(
        self,
        query: str,
        query_type: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Synthesize comprehensive answer from retrieved chunks
        
        Args:
            query: Original user query
            query_type: Query type
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information in the knowledge base to answer this question.",
                'sources': [],
                'confidence': 'low'
            }
        
        # Prepare context from chunks
        context = self._prepare_context(retrieved_chunks)
        
        # Create synthesis prompt based on query type
        if query_type == "GENERAL_KNOWLEDGE":
            system_prompt = """You are an expert astrologer. The user is asking what factors to check for a particular topic.

Provide a comprehensive answer that:
1. Lists ALL relevant factors to check
2. Organizes them by category (houses, planets, dashas, etc.)
3. Explains WHY each factor matters
4. Uses ONLY information from the provided context
5. Cites sources using [Source: page X]

Be thorough and educational."""

        elif query_type == "SPECIFIC_PLACEMENT":
            system_prompt = """You are an expert astrologer. The user has specific placements and wants analysis.

Provide a comprehensive answer that:
1. Analyzes the specific placement mentioned
2. Considers multiple factors (dignity, aspects, house, dasha)
3. Gives a balanced view (positives and negatives)
4. Uses ONLY information from the provided context
5. Cites sources using [Source: page X]

Be specific and practical."""

        else:
            system_prompt = """You are an expert astrologer. Answer the user's question comprehensively.

Guidelines:
1. Use ONLY information from the provided context
2. Be clear and educational
3. Cite sources using [Source: page X]
4. If multiple perspectives exist, mention them"""

        user_prompt = f"""Context from astrological texts:

{context}

---

Question: {query}

Provide a comprehensive answer based on the context above. Remember to cite sources."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract sources mentioned in answer
            sources = self._extract_sources(answer, retrieved_chunks)
            
            logger.info(f"Synthesized answer ({len(answer)} chars, {len(sources)} sources)")
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 'high' if len(retrieved_chunks) >= 5 else 'medium',
                'num_sources': len(sources)
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {str(e)}")
            return {
                'answer': "I encountered an error generating the answer. Please try again.",
                'sources': [],
                'confidence': 'low'
            }
    
    def _prepare_context(self, chunks: List[Dict], max_chunks: int = 10) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            page = chunk.get('metadata', {}).get('page_number', 'unknown')
            content = chunk.get('content', '')
            
            context_parts.append(f"[Source {i+1}, Page {page}]:\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract cited sources from answer"""
        sources = []
        seen_pages = set()
        
        for chunk in chunks:
            page = chunk.get('metadata', {}).get('page_number')
            source_file = chunk.get('metadata', {}).get('source', 'Unknown')
            
            if page and page not in seen_pages:
                # Check if this page is cited in answer
                if f"page {page}" in answer.lower() or f"source {page}" in answer.lower():
                    sources.append({
                        'page': page,
                        'source': source_file,
                        'content_preview': chunk.get('content', '')[:200]
                    })
                    seen_pages.add(page)
        
        return sources
    
    def process_query(
        self,
        query: str,
        retrieval_function,
        max_iterations: int = 3
    ) -> Dict:
        """
        Complete query processing workflow
        
        Args:
            query: User query
            retrieval_function: Function to retrieve chunks (takes query string)
            max_iterations: Maximum refinement iterations
            
        Returns:
            Complete response dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing query: '{query}'")
        logger.info(f"{'='*60}")
        
        # Step 1: Classify query
        classification = self.classify_query(query)
        query_type = classification['type']
        
        # Step 2: Decompose if needed
        sub_queries = self.decompose_query(query, query_type)
        
        # Step 3: Retrieve for each sub-query
        all_chunks = []
        seen_ids = set()
        
        for sub_query in sub_queries[:3]:  # Limit to 3 sub-queries
            logger.info(f"Retrieving for: {sub_query}")
            chunks = retrieval_function(sub_query)
            
            # Add unique chunks
            for chunk in chunks:
                chunk_id = chunk.get('id')
                if chunk_id and chunk_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk_id)
        
        logger.info(f"Retrieved {len(all_chunks)} unique chunks")
        
        # Step 4: Synthesize answer
        result = self.synthesize_answer(query, query_type, all_chunks)
        
        # Add metadata
        result['query_type'] = query_type
        result['sub_queries'] = sub_queries
        result['num_chunks_used'] = len(all_chunks)
        
        return result


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Orchestration Agent (OpenAI)")
    print("="*60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ OPENAI_API_KEY not found in .env")
        print("Please add your OpenAI API key to test the agent")
        exit(0)
    
    try:
        agent = OrchestrationAgent()
        
        # Test queries
        test_queries = [
            "What determines career success in astrology?",
            "I have Mars in 10th house with Saturn aspect, what about my career?",
            "What happens during Sun Mahadasha?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Test classification
            classification = agent.classify_query(query)
            print(f"\nClassification: {classification['type']}")
            
            # Test decomposition
            sub_queries = agent.decompose_query(query, classification['type'])
            print(f"\nDecomposed into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries):
                print(f"  {i+1}. {sq}")
        
        print("\n" + "="*60)
        print("✓ Orchestration Agent Tests Complete")
        print("="*60)
        print("\nNote: Full synthesis testing requires integration with complete_pipeline")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Make sure OPENAI_API_KEY is set in .env")