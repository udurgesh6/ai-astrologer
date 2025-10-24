"""
Streamlit Chat Interface
User-friendly web interface for Astrology PDF Q&A
"""

import streamlit as st
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')
sys.path.insert(0, 'src/agent')

from complete_pipeline import CompletePipeline
from orchestration_agent import OrchestrationAgent


# Page config
st.set_page_config(
    page_title="Astrology PDF Q&A",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.pdf_processed = False


def initialize_system():
    """Initialize the complete pipeline and agent"""
    with st.spinner("Initializing system..."):
        try:
            st.session_state.pipeline = CompletePipeline(
                chunk_size=800,
                chunk_overlap=150,
                use_llm_extraction=False,
                rerank_method="simple"
            )
            st.session_state.agent = OrchestrationAgent()
            return True
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            return False


def process_pdf(pdf_file):
    """Process uploaded PDF"""
    # Save uploaded file
    pdf_path = Path("data/pdfs") / pdf_file.name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Process
    with st.spinner(f"Processing {pdf_file.name}... This may take a few minutes."):
        result = st.session_state.pipeline.process_pdf(
            str(pdf_path),
            extract_entities=True
        )
    
    return result


def search_and_answer(query: str):
    """Search and generate answer"""
    
    # Retrieval function for agent
    def retrieve_chunks(q):
        return st.session_state.pipeline.search(
            q,
            top_k=10,
            use_vector=True,
            use_keyword=True,
            use_graph=True,
            rerank=True
        )
    
    # Process query through agent
    with st.spinner("Thinking..."):
        result = st.session_state.agent.process_query(
            query,
            retrieve_chunks
        )
    
    return result


# Main UI
def main():
    # Header
    st.markdown('<div class="main-header">🔮 Astrology PDF Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your astrology PDFs</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Initialize button
        if st.session_state.pipeline is None:
            if st.button("🚀 Initialize System", use_container_width=True):
                if initialize_system():
                    st.success("✓ System initialized!")
                    st.rerun()
        else:
            st.success("✓ System ready")
        
        st.markdown("---")
        
        # PDF Upload
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose an astrology PDF",
            type=['pdf'],
            help="Upload a PDF document about astrology"
        )
        
        if uploaded_file and st.session_state.pipeline:
            if st.button("📄 Process PDF", use_container_width=True):
                result = process_pdf(uploaded_file)
                
                if result['success']:
                    st.session_state.pdf_processed = True
                    st.success("✓ PDF processed successfully!")
                    
                    # Show stats
                    st.markdown("**Processing Results:**")
                    st.write(f"- Pages: {result['total_pages']}")
                    st.write(f"- Chunks: {result['total_chunks']}")
                    st.write(f"- Entities: {result['total_entities']}")
                    st.write(f"- Graph nodes: {result['graph_nodes']}")
                    st.write(f"- Time: {result['processing_time_seconds']:.1f}s")
                else:
                    st.error(f"✗ Processing failed: {result.get('error')}")
        
        st.markdown("---")
        
        # System stats
        if st.session_state.pipeline and st.session_state.pdf_processed:
            st.subheader("📊 System Statistics")
            
            try:
                stats = st.session_state.pipeline.get_statistics()
                
                st.markdown(f"**Vector Store:** {stats['vector_store']['total_documents']} docs")
                st.markdown(f"**Knowledge Graph:** {stats['knowledge_graph']['total_nodes']} nodes")
                st.markdown(f"**Keyword Index:** {stats['keyword_search']['vocabulary_size']} terms")
            except:
                pass
        
        st.markdown("---")
        
        # Example queries
        st.subheader("💡 Example Queries")
        examples = [
            "What determines career success?",
            "I have Mars in 10th house, what about my career?",
            "What happens in Sun Mahadasha?",
            "How to predict marriage timing?",
            "What are wealth indicators?"
        ]
        
        for ex in examples:
            if st.button(f"📝 {ex}", use_container_width=True, key=ex):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
    
    # Main chat area
    if st.session_state.pipeline is None:
        st.info("👈 Click 'Initialize System' in the sidebar to get started")
        return
    
    if not st.session_state.pdf_processed:
        st.info("👈 Upload and process a PDF to start asking questions")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** Page {source['page']}")
                            st.caption(source['content_preview'][:200] + "...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about astrology..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing..."):
                result = search_and_answer(prompt)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"🎯 Type: {result.get('query_type', 'Unknown')}")
            with col2:
                st.caption(f"📊 Sources: {result.get('num_sources', 0)}")
            with col3:
                st.caption(f"✓ Confidence: {result.get('confidence', 'unknown')}")
            
            # Show sources
            if result.get('sources'):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result['sources']):
                        st.markdown(f"**Source {i+1}:** Page {source['page']}")
                        st.caption(source['content_preview'][:200] + "...")
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result.get('sources', [])
        })


if __name__ == "__main__":
    main()