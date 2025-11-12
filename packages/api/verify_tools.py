#!/usr/bin/env python3
"""
Simple Tool Verification Script
================================
Check which tools are configured and functional without requiring full setup.
"""

import os
import sys
from pathlib import Path

# Load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"  📄 Loaded .env file: {env_file}\n")
    except ImportError:
        # Try to load manually if python-dotenv not installed
        print(f"  ⚠️  .env file found but python-dotenv not installed\n")
        print(f"      Install: pip install python-dotenv\n")
        print(f"      Or manually load .env file\n")
        # Manual loading as fallback
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
            print(f"  ✅ Manually loaded .env file\n")
        except Exception as e:
            print(f"  ❌ Error loading .env: {e}\n")
else:
    print(f"  ⚠️  .env file not found at: {env_file}\n")

print("\n" + "="*70)
print("  🔍 TOOL VERIFICATION - ML DOCS COPILOT")
print("="*70 + "\n")


def check_env_var(name, description):
    """Check if environment variable is set."""
    value = os.getenv(name)
    if value:
        masked = f"{value[:8]}..." if len(value) > 8 else value
        print(f"  ✅ {name:30} SET        {masked}")
        return True
    else:
        print(f"  ⚠️  {name:30} NOT SET    (Optional)")
        return False


def check_python_module(module_name, description):
    """Check if Python module is available."""
    try:
        __import__(module_name)
        print(f"  ✅ {description:30} Available")
        return True
    except ImportError:
        print(f"  ❌ {description:30} Not Installed")
        return False


def check_file_exists(path, description):
    """Check if file or directory exists."""
    if Path(path).exists():
        print(f"  ✅ {description:30} Found")
        return True
    else:
        print(f"  ⚠️  {description:30} Not Found")
        return False


def test_retrieval_system():
    """Test if retrieval system is accessible."""
    print("\n📚 Testing Retrieval System")
    print("-" * 70)
    
    # Check database
    data_dir = Path("data")
    qdrant_dir = data_dir / "qdrant"
    
    db_exists = check_file_exists(str(qdrant_dir), "Qdrant Database")
    
    # Check if we can import retrieval
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from api.retrieval import HybridRetriever
        print("  ✅ HybridRetriever        Importable")
        
        if db_exists:
            try:
                retriever = HybridRetriever()
                print("  ✅ HybridRetriever        Initialized")
                return True
            except Exception as e:
                print(f"  ❌ HybridRetriever        Error: {str(e)[:50]}")
                return False
        else:
            print("  ⚠️  HybridRetriever        Database not found")
            return False
    except ImportError as e:
        print(f"  ❌ HybridRetriever        Import Error: {str(e)[:50]}")
        return False


def test_web_search():
    """Test web search configuration."""
    print("\n🌐 Testing Web Search")
    print("-" * 70)
    
    tavily_key = check_env_var("TAVILY_API_KEY", "Tavily API Key")
    
    if tavily_key:
        tavily_available = check_python_module("langchain_community.tools.tavily_search", "Tavily Search Tool")
        if tavily_available:
            print("  ✅ Web Search              Configured & Ready")
            return True
        else:
            print("  ⚠️  Web Search              API key set but package missing")
            print("      Install: pip install langchain-community")
            return False
    else:
        print("  ⚠️  Web Search              Not Configured")
        print("      Get key: https://tavily.com")
        print("      Add to .env: TAVILY_API_KEY=your_key")
        return False


def test_code_execution():
    """Test code execution configuration."""
    print("\n💻 Testing Code Execution")
    print("-" * 70)
    
    e2b_key = check_env_var("E2B_API_KEY", "E2B API Key")
    
    if e2b_key:
        # Check for e2b-code-interpreter package
        e2b_code_interp = check_python_module("e2b_code_interpreter", "E2B Code Interpreter")
        e2b_base = check_python_module("e2b", "E2B Base Package")
        
        if e2b_code_interp and e2b_base:
            print("  ✅ Code Execution          Configured & Ready")
            return True
        else:
            print("  ⚠️  Code Execution          API key set but packages missing")
            if not e2b_base:
                print("      Missing: pip install e2b")
            if not e2b_code_interp:
                print("      Missing: pip install e2b-code-interpreter")
            return False
    else:
        print("  ⚠️  Code Execution          Not Configured")
        print("      Get key: https://e2b.dev")
        print("      Add to .env: E2B_API_KEY=your_key")
        return False


def test_code_validation():
    """Test code validation (always available)."""
    print("\n✔️  Testing Code Validation")
    print("-" * 70)
    
    try:
        import ast
        # Test with valid code
        ast.parse("x = 42")
        print("  ✅ Code Validation         Working (AST parser)")
        return True
    except Exception as e:
        print(f"  ❌ Code Validation         Error: {str(e)[:50]}")
        return False


def test_llm_configuration():
    """Test LLM/Vertex AI configuration."""
    print("\n🤖 Testing LLM Configuration")
    print("-" * 70)
    
    gcp_project = check_env_var("GOOGLE_CLOUD_PROJECT", "GCP Project ID")
    gcp_location = check_env_var("GOOGLE_CLOUD_LOCATION", "GCP Location")
    gcp_creds = check_env_var("GOOGLE_APPLICATION_CREDENTIALS", "GCP Credentials")
    
    langchain_available = check_python_module("langchain_google_vertexai", "LangChain Vertex AI")
    
    if gcp_project and (gcp_creds or check_file_exists("gcp-credentials.json", "GCP Credentials File")):
        if langchain_available:
            print("  ✅ LLM Configuration       Configured & Ready")
            return True
        else:
            print("  ⚠️  LLM Configuration       Credentials set but package missing")
            print("      Install: pip install langchain-google-vertexai")
            return False
    else:
        print("  ⚠️  LLM Configuration       Not Fully Configured")
        print("      Required: GOOGLE_CLOUD_PROJECT")
        print("      Required: GOOGLE_APPLICATION_CREDENTIALS or gcp-credentials.json")
        return False


def main():
    """Run all verification checks."""
    
    # Environment Configuration
    print("🔐 Environment Configuration")
    print("-" * 70)
    
    check_env_var("GOOGLE_CLOUD_PROJECT", "GCP Project ID")
    check_env_var("GOOGLE_CLOUD_LOCATION", "GCP Location")
    check_env_var("GOOGLE_APPLICATION_CREDENTIALS", "GCP Credentials Path")
    check_env_var("TAVILY_API_KEY", "Tavily API Key")
    check_env_var("E2B_API_KEY", "E2B API Key")
    
    # Test each component
    results = {}
    
    results["Retrieval"] = test_retrieval_system()
    results["Web Search"] = test_web_search()
    results["Code Execution"] = test_code_execution()
    results["Code Validation"] = test_code_validation()
    results["LLM"] = test_llm_configuration()
    
    # Summary
    print("\n" + "="*70)
    print("  📊 VERIFICATION SUMMARY")
    print("="*70 + "\n")
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    for tool, status in results.items():
        icon = "✅" if status else "⚠️"
        print(f"  {icon} {tool:25} {'Working' if status else 'Not Configured/Broken'}")
    
    print(f"\n  Status: {working}/{total} tools working")
    
    # Recommendations
    print("\n" + "="*70)
    print("  💡 RECOMMENDATIONS")
    print("="*70 + "\n")
    
    if working == total:
        print("  ✅ All tools are configured and working!")
        print("     Your system is fully operational.")
    elif working >= 3:
        print("  ⚠️  Core tools are working.")
        if not results["Web Search"]:
            print("     • Configure TAVILY_API_KEY for web search")
        if not results["Code Execution"]:
            print("     • Configure E2B_API_KEY for code execution")
        print("     • Optional tools can be configured later")
    else:
        print("  ❌ Critical tools need attention:")
        if not results["Retrieval"]:
            print("     • Set up database: python -m ingest.ingest_all")
        if not results["LLM"]:
            print("     • Configure GCP credentials for LLM")
        if not results["Code Validation"]:
            print("     • Check Python installation")
    
    print("\n" + "="*70 + "\n")
    
    # Return exit code
    return 0 if working >= 3 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Verification failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


