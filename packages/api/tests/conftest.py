"""Pytest configuration and fixtures."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_conversation_id():
    """Generate a test conversation ID."""
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "How do I create a DataLoader in PyTorch?"


@pytest.fixture
def mock_tavily_search():
    """Mock Tavily search responses."""
    with patch('agent.tools.web_tools.TavilySearchResults') as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value = [
            {
                "title": "PyTorch Documentation",
                "url": "https://pytorch.org/docs",
                "content": "Official PyTorch documentation for DataLoader...",
                "score": 0.95
            },
            {
                "title": "PyTorch Tutorial",
                "url": "https://pytorch.org/tutorials",
                "content": "Learn how to use DataLoader in PyTorch...",
                "score": 0.88
            }
        ]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_e2b_sandbox():
    """Mock E2B sandbox for code execution."""
    with patch('agent.tools.code_executor_tool.Sandbox') as mock:
        mock_execution = Mock()
        mock_execution.error = None
        mock_execution.logs = Mock()
        mock_execution.logs.stdout = [Mock(line="Result: 4\n")]
        mock_execution.logs.stderr = []
        
        mock_sandbox = Mock()
        mock_sandbox.__enter__ = Mock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = Mock(return_value=None)
        mock_sandbox.run_code.return_value = mock_execution
        
        mock.return_value = mock_sandbox
        yield mock


@pytest.fixture
def mock_retriever():
    """Mock HybridRetriever for testing."""
    from unittest.mock import MagicMock
    from api.retrieval import RetrievalResult
    
    mock = MagicMock()
    
    # Create mock results
    mock_results = [
        RetrievalResult(
            chunk_id="chunk_1",
            content="PyTorch DataLoader is used to load data in batches...",
            score=0.95,
            metadata={
                "vendor": "PyTorch",
                "source_url": "https://pytorch.org/docs/stable/data.html",
                "heading_path": "PyTorch > Data Loading",
                "doc_type": "documentation"
            }
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            content="The DataLoader class provides batching, shuffling, and multiprocessing...",
            score=0.88,
            metadata={
                "vendor": "PyTorch",
                "source_url": "https://pytorch.org/docs/stable/data.html",
                "heading_path": "PyTorch > DataLoader",
                "doc_type": "documentation"
            }
        )
    ]
    
    mock.retrieve.return_value = mock_results
    
    return mock


@pytest.fixture
def mock_environment_variables():
    """Set mock environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "TAVILY_API_KEY": "test-tavily-key",
        "E2B_API_KEY": "test-e2b-key"
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_vertexai():
    """Mock Vertex AI for testing."""
    with patch('agent.orchestrator.vertexai') as mock:
        mock.init = Mock()
        yield mock


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from langchain_core.messages import AIMessage
    
    return AIMessage(
        content="PyTorch DataLoader is a powerful utility for loading data in batches. Here's how to use it:\n\n```python\nimport torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\n# Create dataset\ndataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))\n\n# Create DataLoader\ndataloader = DataLoader(dataset, batch_size=32, shuffle=True)\n```\n\nThe DataLoader provides automatic batching, shuffling, and multiprocessing support."
    )


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


