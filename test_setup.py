"""
Setup Verification Script
Verifies that all components are properly installed and configured
"""

import sys
from pathlib import Path
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        ("fitz", "PyMuPDF"),
        ("chromadb", "chromadb"),
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
        ("anthropic", "anthropic"),
        ("langchain", "langchain"),
        ("streamlit", "streamlit"),
        ("networkx", "networkx"),
    ]
    
    all_installed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (run: pip install {package_name})")
            all_installed = False
    
    return all_installed

def check_env_file():
    """Check if .env file exists and has required keys"""
    print("\nChecking environment configuration...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("✗ .env file not found")
        print("  → Copy .env.template to .env and add your API keys")
        return False
    
    print("✓ .env file exists")
    
    # Check for API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    keys_found = True
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✓ OPENAI_API_KEY configured")
    else:
        print("✗ OPENAI_API_KEY not configured")
        keys_found = False
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        print("✓ ANTHROPIC_API_KEY configured")
    else:
        print("⚠ ANTHROPIC_API_KEY not configured (needed for Sprint 5)")
    
    return keys_found

def check_directories():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "data/pdfs",
        "data/chroma_db",
        "data/processed",
        "src/pdf_processing",
        "src/retrieval",
        "src/agent",
        "src/ui",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (creating...)")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return True  # We create them if missing

def check_modules():
    """Check if custom modules can be imported"""
    print("\nChecking custom modules...")
    
    # Add src to path
    sys.path.insert(0, str(Path("src/pdf_processing")))
    sys.path.insert(0, str(Path("src/retrieval")))
    
    modules = [
        ("pdf_extractor", "PDFExtractor"),
        ("text_chunker", "TextChunker"),
        ("vector_store", "VectorStore"),
    ]
    
    all_imported = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except ImportError as e:
            print(f"✗ {module_name}.{class_name} (ImportError: {str(e)})")
            all_imported = False
        except AttributeError as e:
            print(f"✗ {module_name}.{class_name} (AttributeError: {str(e)})")
            all_imported = False
    
    return all_imported

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\nTesting OpenAI API connection...")
    
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "your_openai_api_key_here":
            print("⚠ Skipping OpenAI test (API key not configured)")
            return True
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input="Test connection"
        )
        
        if response.data[0].embedding:
            print("✓ OpenAI API connection successful")
            return True
        else:
            print("✗ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ OpenAI API connection failed: {str(e)}")
        return False

def check_sample_pdf():
    """Check if sample PDF exists"""
    print("\nChecking for sample PDF...")
    
    pdf_dir = Path("data/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if pdfs:
        print(f"✓ Found {len(pdfs)} PDF(s) in data/pdfs/")
        for pdf in pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"  - {pdf.name} ({size_mb:.2f} MB)")
        return True
    else:
        print("⚠ No PDFs found in data/pdfs/")
        print("  → Add a PDF to test the pipeline")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n" + "="*60)
    print("Running Quick Functionality Test")
    print("="*60)
    
    try:
        # Add src to path for imports
        sys.path.insert(0, str(Path("src/pdf_processing")))
        from text_chunker import TextChunker, Chunk
        
        print("\nTesting TextChunker...")
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
        
        test_text = """
        Astrology is an ancient science. It studies celestial influences.
        
        The birth chart shows planetary positions. Each planet has significance.
        
        Mars represents energy and courage. Sun represents soul and authority.
        """
        
        chunks = chunker.chunk_by_paragraphs(test_text, {"test": True})
        
        if chunks:
            print(f"✓ Created {len(chunks)} test chunks")
            print(f"  First chunk: {chunks[0].content[:50]}...")
            return True
        else:
            print("✗ Failed to create chunks")
            return False
            
    except Exception as e:
        print(f"✗ Quick test failed: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("Astrology PDF Q&A - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Directories", check_directories),
        ("Custom Modules", check_modules),
        ("Sample PDF", check_sample_pdf),
        ("Quick Test", run_quick_test),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"✗ {name} check failed with error: {str(e)}")
            results[name] = False
    
    # Optional: Test OpenAI if configured
    if results.get("Environment File"):
        results["OpenAI Connection"] = test_openai_connection()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✓ All checks passed! You're ready to start development.")
        print("\nNext steps:")
        print("1. Add a PDF to data/pdfs/")
        print("2. Run: python pdf_pipeline.py")
        print("3. Test retrieval with sample queries")
    elif passed >= total - 2:
        print("\n⚠ Most checks passed. Review warnings above.")
        print("\nYou can proceed, but some features may not work.")
    else:
        print("\n✗ Several checks failed. Please fix issues above.")
        print("\nRefer to README.md for setup instructions.")
    
    print("\n" + "="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)