"""
Orchestration Agent 2 - Domain-Aware
Intelligent query processing with domain-based routing
"""

from typing import List, Dict, Optional
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationAgent2:
    """
    Enhanced intelligent agent with domain awareness:
    1. Understands and classifies queries
    2. Suggests relevant domains to search
    3. Decomposes complex questions
    4. Synthesizes comprehensive answers
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize orchestration agent
        
        Args:
            api_key: OpenAI API key (optional, uses env if not provided)
            model: OpenAI model to use
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Add it to .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        logger.info(f"Orchestration agent v2 initialized with {model}")
    
    def classify_query_with_domains(self, query: str, available_domains: List[str]) -> Dict:
        """
        Classify query and suggest relevant domains
        
        Args:
            query: User query
            available_domains: List of available domain names
            
        Returns:
            Dictionary with classification and suggested domains
        """
        domains_list = ", ".join(available_domains)
        
        prompt = f"""Analyze this astrology query and determine:
1. The query type
2. Which domains are most relevant

Query: "{query}"

Available domains: {domains_list}

Query Types:
- GENERAL_KNOWLEDGE: What factors to check (e.g., "What determines career success?")
- SPECIFIC_PLACEMENT: About specific placements (e.g., "I have Mars in 10th house")
- COMPARISON: Comparing things (e.g., "What's better for career: Sun or Mars?")
- TIMING: About timing/dasha (e.g., "What happens in Sun Mahadasha?")
- REMEDY: About remedies/solutions (e.g., "How to fix weak Jupiter?")
- GENERAL_QUESTION: General astrology question

Respond in this format:
TYPE: [query type]
DOMAINS: [comma-separated list of 1-3 most relevant domains, ALWAYS include 'base' if available]
REASON: [brief reason for domain selection]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse response
            query_type = "GENERAL_QUESTION"
            suggested_domains = ['base'] if 'base' in available_domains else []
            reason = ""
            
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith("TYPE:"):
                    query_type = line.replace("TYPE:", "").strip().upper()
                elif line.startswith("DOMAINS:"):
                    domains_text = line.replace("DOMAINS:", "").strip()
                    suggested_domains = [d.strip() for d in domains_text.split(',')]
                    # Filter to only available domains
                    suggested_domains = [d for d in suggested_domains if d in available_domains]
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            
            # Ensure base is included if available
            if 'base' in available_domains and 'base' not in suggested_domains:
                suggested_domains.insert(0, 'base')
            
            logger.info(f"Query type: {query_type}, Suggested domains: {suggested_domains}")
            
            return {
                'type': query_type,
                'query': query,
                'suggested_domains': suggested_domains,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                'type': 'GENERAL_QUESTION',
                'query': query,
                'suggested_domains': ['base'] if 'base' in available_domains else available_domains[:1],
                'reason': 'Error in classification'
            }
    
    def decompose_query(self, query: str, query_type: str) -> List[str]:
        """
        Decompose complex query into sub-queries
        
        Args:
            query: User query
            query_type: Query type from classification
            
        Returns:
            List of sub-queries
        """
        if query_type in ["GENERAL_KNOWLEDGE", "SPECIFIC_PLACEMENT"]:
            prompt = f"""Break down this astrology question into specific aspects to research:

Question: "{query}"
Type: {query_type}

List 2-4 specific aspects that should be researched. Return as a numbered list.

Example for "What determines career success?":
1. 10th house and its lord
2. Saturn position and strength
3. Sun placement
4. Current dasha period

Now for the given question:"""

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
                    if line and (line[0].isdigit() or line.startswith('-')):
                        clean = line.lstrip('0123456789.-) ').strip()
                        if clean:
                            sub_queries.append(clean)
                
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
                return sub_queries if sub_queries else [query]
                
            except Exception as e:
                logger.error(f"Decomposition failed: {str(e)}")
                return [query]
        else:
            return [query]
    
    def synthesize_answer(
        self,
        query: str,
        query_type: str,
        retrieved_chunks: List[Dict],
        domains_searched: List[str]
    ) -> Dict:
        """
        Synthesize comprehensive answer from retrieved chunks
        
        Args:
            query: Original user query
            query_type: Query type
            retrieved_chunks: Retrieved context chunks
            domains_searched: List of domains that were searched
            
        Returns:
            Dictionary with answer and metadata
        """
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information in the knowledge base to answer this question. Try rephrasing or check if the relevant domain has been processed.",
                'sources': [],
                'confidence': 'low',
                'domains_used': domains_searched
            }
        
        # Prepare context from chunks
        context = self._prepare_context(retrieved_chunks)
        
        # Create synthesis prompt based on query type
        system_prompts = {
            "GENERAL_KNOWLEDGE": """You are an expert Vedic astrologer. The user is asking what factors to check for a particular topic.

Provide a comprehensive answer that:
1. Lists ALL relevant factors to check
2. Organizes them by category (houses, planets, dashas, etc.)
3. Explains WHY each factor matters
4. Uses ONLY information from the provided context
5. Cites sources using [Page X]

Be thorough and educational.""",
            
            "SPECIFIC_PLACEMENT": """You are an expert Vedic astrologer. The user has specific placements and wants analysis.

Provide a comprehensive answer that:
1. Analyzes the specific placement mentioned
2. Considers multiple factors (dignity, aspects, house, dasha)
3. Gives a balanced view (positives and negatives)
4. Uses ONLY information from the provided context
5. Cites sources using [Page X]

Be specific and practical.""",
            
            "TIMING": """You are an expert Vedic astrologer. The user is asking about timing or dasha periods.

Provide a comprehensive answer that:
1. Explains the relevant timing considerations
2. Discusses dasha/transit effects if applicable
3. Gives timeframes if mentioned in context
4. Uses ONLY information from the provided context
5. Cites sources using [Page X]

Be specific about timing indicators.""",
            
            "REMEDY": """You are an expert Vedic astrologer. The user is asking about remedies or solutions.

Provide a comprehensive answer that:
1. Lists relevant remedies from the context
2. Explains why each remedy works
3. Gives practical implementation advice
4. Uses ONLY information from the provided context
5. Cites sources using [Page X]

Be practical and specific."""
        }
        
        system_prompt = system_prompts.get(query_type, """You are an expert Vedic astrologer. Answer the user's question comprehensively.

Guidelines:
1. Use ONLY information from the provided context
2. Be clear and educational
3. Cite sources using [Page X]
4. If multiple perspectives exist, mention them
5. Acknowledge if information is limited""")

        # Add domain context
        domains_context = f"\nNote: This answer is based on information from the following domains: {', '.join(domains_searched)}\n"
        
        user_prompt = f"""Context from astrological texts:{domains_context}

{context}

---

Question: {query}

Provide a comprehensive answer based on the context above. Remember to cite sources by page number."""

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
                'num_sources': len(sources),
                'domains_used': domains_searched
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {str(e)}")
            return {
                'answer': "I encountered an error generating the answer. Please try again.",
                'sources': [],
                'confidence': 'low',
                'domains_used': domains_searched
            }
    
    def _prepare_context(self, chunks: List[Dict], max_chunks: int = 10) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            page = chunk.get('metadata', {}).get('page_number', 'unknown')
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            domain = chunk.get('domain', 'unknown')
            content = chunk.get('content', '')
            
            context_parts.append(
                f"[Source {i+1} - {source}, Page {page}, Domain: {domain}]:\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract cited sources from answer"""
        sources = []
        seen_pages = set()
        
        for chunk in chunks:
            page = chunk.get('metadata', {}).get('page_number')
            source_file = chunk.get('metadata', {}).get('source', 'Unknown')
            domain = chunk.get('domain', 'unknown')
            
            if page and page not in seen_pages:
                # Check if this page is cited in answer
                if f"page {page}" in answer.lower() or f"[page {page}]" in answer.lower():
                    sources.append({
                        'page': page,
                        'source': source_file,
                        'domain': domain,
                        'content_preview': chunk.get('content', '')[:200]
                    })
                    seen_pages.add(page)
        
        return sources
    
    def process_query(
        self,
        query: str,
        retrieval_function,
        available_domains: Optional[List[str]] = None
    ) -> Dict:
        """
        Complete query processing workflow with domain awareness
        
        Args:
            query: User query
            retrieval_function: Function to retrieve chunks (takes query and domains)
            available_domains: List of available domains
            
        Returns:
            Complete response dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing query: '{query}'")
        logger.info(f"{'='*60}")
        
        # Step 1: Classify query and suggest domains
        if available_domains:
            classification = self.classify_query_with_domains(query, available_domains)
            suggested_domains = classification['suggested_domains']
        else:
            classification = {'type': 'GENERAL_QUESTION', 'query': query}
            suggested_domains = None
        
        query_type = classification['type']
        
        # Step 2: Decompose if needed
        sub_queries = self.decompose_query(query, query_type)
        
        # Step 3: Retrieve for each sub-query
        all_chunks = []
        seen_ids = set()
        
        for sub_query in sub_queries[:3]:  # Limit to 3 sub-queries
            logger.info(f"Retrieving for: {sub_query}")
            
            # Call retrieval with suggested domains
            if hasattr(retrieval_function, '__code__') and 'domains' in retrieval_function.__code__.co_varnames:
                chunks = retrieval_function(sub_query, suggested_domains)
            else:
                chunks = retrieval_function(sub_query)
            
            # Add unique chunks
            for chunk in chunks:
                chunk_id = chunk.get('id')
                if chunk_id and chunk_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk_id)
        
        logger.info(f"Retrieved {len(all_chunks)} unique chunks")
        
        # Step 4: Synthesize answer
        result = self.synthesize_answer(
            query,
            query_type,
            all_chunks,
            suggested_domains or ['unknown']
        )
        
        # Add metadata
        result['query_type'] = query_type
        result['sub_queries'] = sub_queries
        result['num_chunks_used'] = len(all_chunks)
        result['suggested_domains'] = suggested_domains
        if 'reason' in classification:
            result['domain_selection_reason'] = classification['reason']
        
        return result


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Orchestration Agent 2 (Domain-Aware)")
    print("="*60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ OPENAI_API_KEY not found in .env")
        print("Please add your OpenAI API key to test the agent")
        exit(0)
    
    try:
        agent = OrchestrationAgent2()
        
        # Available domains
        available_domains = ['base', 'children', 'marriage', 'career', 'remedies', 'wealth']
        
        # Test queries
        test_queries = [
            "When will I have children?",
            "What determines career success in astrology?",
            "Remedies for weak Mars?",
            "I have Mars in 10th house with Saturn aspect, what about my career?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Test classification with domain suggestion
            classification = agent.classify_query_with_domains(query, available_domains)
            print(f"\nType: {classification['type']}")
            print(f"Suggested Domains: {classification['suggested_domains']}")
            print(f"Reason: {classification.get('reason', 'N/A')}")
            
            # Test decomposition
            sub_queries = agent.decompose_query(query, classification['type'])
            print(f"\nDecomposed into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries):
                print(f"  {i+1}. {sq}")
        
        print("\n" + "="*60)
        print("✓ Orchestration Agent 2 Tests Complete")
        print("="*60)
        print("\nNote: Full synthesis testing requires integration with complete_pipeline3")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Make sure OPENAI_API_KEY is set in .env")