"""
Vector Store Module
Handles storage and retrieval of document embeddings using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
from pathlib import Path
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

os.environ['ANONYMIZED_TELEMETRY'] = 'False'

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and similarity search"""
    
    def __init__(
        self,
        collection_name: str = "astrology_docs",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Astrology PDF document chunks"}
        )
        
        logger.info(f"Vector store initialized: {self.collection_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Generate embeddings one by one for this batch
                for text in batch:
                    try:
                        emb = self.generate_embedding(text)
                        all_embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Failed to embed text: {str(e2)}")
                        # Add zero vector as placeholder
                        all_embeddings.append([0.0] * 3072)  # Default dimension for text-embedding-3-large
        
        return all_embeddings
    
    def add_documents(
        self,
        chunks: List,  # List of Chunk objects
        batch_size: int = 100
    ) -> None:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of Chunk objects from TextChunker
            batch_size: Batch size for processing
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size)
        
        # Add to collection in batches
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            try:
                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                logger.info(f"Added batch {i//batch_size + 1} to collection")
            except Exception as e:
                logger.error(f"Error adding batch to collection: {str(e)}")
                raise
        
        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"page_number": 5})
            
        Returns:
            List of results with content and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict if filter_dict else None
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def search_with_score(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search with similarity score filtering
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of results above threshold
        """
        results = self.similarity_search(query, k)
        
        if score_threshold:
            # Convert distance to similarity score (1 - distance)
            filtered_results = []
            for result in results:
                if result['distance'] is not None:
                    similarity = 1 - result['distance']
                    result['similarity_score'] = similarity
                    if similarity >= score_threshold:
                        filtered_results.append(result)
            return filtered_results
        
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'total_documents': count,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def delete_by_source(self, source: str) -> None:
        """
        Delete all documents from a specific source
        
        Args:
            source: Source document name
        """
        try:
            # Query to get all IDs with this source
            results = self.collection.get(
                where={"source": source}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents from source: {source}")
            else:
                logger.info(f"No documents found for source: {source}")
                
        except Exception as e:
            logger.error(f"Error deleting by source: {str(e)}")
            raise
    
    def update_document(self, chunk_id: str, new_content: str, new_metadata: Dict) -> None:
        """
        Update a document in the collection
        
        Args:
            chunk_id: ID of the chunk to update
            new_content: New content
            new_metadata: New metadata
        """
        try:
            # Generate new embedding
            new_embedding = self.generate_embedding(new_content)
            
            # Update in collection
            self.collection.update(
                ids=[chunk_id],
                embeddings=[new_embedding],
                documents=[new_content],
                metadatas=[new_metadata]
            )
            logger.info(f"Updated document: {chunk_id}")
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Make sure you have OPENAI_API_KEY in .env file
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="test_astrology",
        persist_directory="./data/test_chroma"
    )
    
    # Sample chunks (normally from TextChunker)
    from text_chunker import Chunk
    
    sample_chunks = [
        Chunk(
            content="Mars in the 10th house gives strong career ambitions and leadership qualities. The native becomes very determined and works hard for professional success.",
            metadata={"source": "test.pdf", "page_number": 1, "topic": "career"}
        ),
        Chunk(
            content="Sun in the 10th house is an excellent placement for career. It gives authority, recognition, and success in profession. The person may work in government or leadership roles.",
            metadata={"source": "test.pdf", "page_number": 1, "topic": "career"}
        ),
        Chunk(
            content="Venus in the 7th house is good for marriage and partnerships. It brings harmony, love, and understanding in relationships.",
            metadata={"source": "test.pdf", "page_number": 2, "topic": "marriage"}
        )
    ]
    
    # Add documents
    print("Adding sample chunks...")
    vector_store.add_documents(sample_chunks)
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    
    # Search
    print("\n--- Testing Search ---")
    query = "What about career and professional success?"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result['content'][:100]}...")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']:.4f}\n")