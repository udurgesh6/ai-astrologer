"""
Streamlit Chat Interface v2 - Domain-Based Knowledge Base
Simple Q&A interface - auto-loads pre-processed knowledge bases
"""

import streamlit as st
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'src/pdf_processing')
sys.path.insert(0, 'src/retrieval')
sys.path.insert(0, 'src/agent')

from complete_pipeline2 import CompletePipeline2
from domain_definitions import DOMAIN_DEFINITIONS
from orchestration_agent2 import OrchestrationAgent2


# Page config
st.set_page_config(
    page_title="Astrology Q&A",
    page_icon="🔮",
    layout="wide"
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
    .domain-badge {
        display: inline-block;
        background-color: #e8f4f8;
        color: #1f77b4;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state and auto-load knowledge base
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pipeline = None
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.error_message = None

# Auto-initialize on first run
if not st.session_state.initialized:
    with st.spinner("🔄 Loading knowledge base..."):
        try:
            # Initialize pipeline
            st.session_state.pipeline = CompletePipeline2(
                base_dir="data2",
                chunk_size=800,
                chunk_overlap=150,
                use_llm_extraction=False,
                rerank_method="simple"
            )
            
            # Initialize agent
            st.session_state.agent = OrchestrationAgent2()
            
            # Check if domains exist
            available_domains = st.session_state.pipeline.list_domains()
            
            if not available_domains:
                st.session_state.error_message = "no_domains"
            else:
                st.session_state.initialized = True
                
        except Exception as e:
            st.session_state.error_message = f"init_error: {str(e)}"


def search_and_answer(query: str):
    """Search and generate answer using domain-based routing"""
    
    # Retrieval function for agent
    def retrieve_chunks(q, domains=None):
        return st.session_state.pipeline.search(
            q,
            top_k=10,
            domains=domains,
            use_vector=True,
            use_keyword=True,
            use_graph=True,
            rerank=True
        )
    
    # Get available domains
    available_domains = st.session_state.pipeline.list_domains()
    
    # Process query through agent
    result = st.session_state.agent.process_query(
        query,
        retrieve_chunks,
        available_domains=available_domains
    )
    
    return result


def display_domain_badge(domain: str):
    """Display a styled domain badge"""
    domain_info = DOMAIN_DEFINITIONS.get(domain, {})
    description = domain_info.get('description', domain)
    return f'<span class="domain-badge" title="{description}">{domain.upper()}</span>'


# Main UI
def main():
    # Header
    st.markdown('<div class="main-header">🔮 Astrology Q&A</div>', unsafe_allow_html=True)
    
    # Handle errors
    if st.session_state.error_message:
        if st.session_state.error_message == "no_domains":
            st.error("❌ No knowledge base found!")
            st.info("""
            **First time setup required:**
            
            1. Process your PDFs using `complete_pipeline2.py`:
            ```python
            from complete_pipeline2 import CompletePipeline2
            
            pipeline = CompletePipeline2(base_dir="data2")
            
            pdf_configs = [
                {'path': 'data/pdfs/your-base-file.pdf', 'domain': 'base'},
                {'path': 'data/pdfs/your-other-file.pdf'},
            ]
            
            pipeline.process_pdfs(pdf_configs)
            ```
            
            2. Refresh this page after processing
            """)
            
            if st.button("🔄 Retry Loading"):
                st.session_state.initialized = False
                st.session_state.error_message = None
                st.rerun()
            return
        else:
            st.error(f"❌ Initialization Error: {st.session_state.error_message}")
            if st.button("🔄 Retry"):
                st.session_state.initialized = False
                st.session_state.error_message = None
                st.rerun()
            return
    
    if not st.session_state.initialized:
        st.info("⏳ Loading knowledge base...")
        return
    
    # Show available domains in sidebar
    with st.sidebar:
        st.header("📚 Knowledge Base")
        
        try:
            available_domains = st.session_state.pipeline.list_domains()
            st.write(f"**{len(available_domains)} domains loaded:**")
            
            for domain in available_domains:
                domain_info = DOMAIN_DEFINITIONS.get(domain, {})
                description = domain_info.get('description', 'No description')
                always_include = "⭐" if domain_info.get('always_include', False) else "•"
                
                with st.expander(f"{always_include} {domain.upper()}", expanded=False):
                    st.caption(description)
        except:
            pass
        
        st.markdown("---")
        
        # Example queries
        st.header("💡 Try These")
        examples = [
            "When will I have children?",
            "What is the best time for marriage?",
            "How to improve career prospects?",
            "Remedies for financial problems",
            "Study abroad possibilities?",
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
        
        st.markdown("---")
        
        # Statistics
        try:
            stats = st.session_state.pipeline.get_statistics()
            st.caption(f"📊 {stats['total_domains']} domains")
            total_vectors = sum(d.get('vector_count', 0) for d in stats['domains'].values())
            st.caption(f"📑 {total_vectors:,} documents")
        except:
            pass
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant":
                # Show domains searched
                if "domains_searched" in message:
                    domains_html = " ".join([display_domain_badge(d) for d in message["domains_searched"]])
                    st.markdown(f"**Searched:** {domains_html}", unsafe_allow_html=True)
                
                # Show sources
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(
                                f'<div class="source-box">'
                                f'<strong>Source {i+1}:</strong> {source.get("source", "Unknown")} - Page {source["page"]}<br>'
                                f'<small>{source["content_preview"][:150]}...</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about astrology..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                try:
                    result = search_and_answer(prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Get domains that were searched
                    suggested_domains = result.get('suggested_domains', [])
                    if suggested_domains:
                        domains_html = " ".join([display_domain_badge(d) for d in suggested_domains])
                        st.markdown(f"**Searched:** {domains_html}", unsafe_allow_html=True)
                    
                    # Show sources
                    if result.get('sources'):
                        with st.expander(f"📚 View {len(result['sources'])} Sources"):
                            for i, source in enumerate(result['sources']):
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<strong>Source {i+1}:</strong> {source.get("source", "Unknown")} - Page {source["page"]}<br>'
                                    f'<small>{source["content_preview"][:150]}...</small>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', []),
                        "domains_searched": suggested_domains
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()