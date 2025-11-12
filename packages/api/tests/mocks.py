"""Mock implementations for external dependencies."""

from typing import List, Dict, Any, Optional
from unittest.mock import Mock
from dataclasses import dataclass


@dataclass
class MockRetrievalResult:
    """Mock retrieval result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class MockHybridRetriever:
    """Mock implementation of HybridRetriever."""
    
    def __init__(self):
        self.mock_data = {
            "pytorch dataloader": [
                MockRetrievalResult(
                    chunk_id="pytorch_dl_1",
                    content="PyTorch DataLoader is a utility class that provides an iterable over a dataset. It supports automatic batching, shuffling, and multiprocessing.",
                    score=0.95,
                    metadata={
                        "vendor": "PyTorch",
                        "source_url": "https://pytorch.org/docs/stable/data.html",
                        "heading_path": "PyTorch > Data Loading",
                        "doc_type": "documentation"
                    }
                ),
                MockRetrievalResult(
                    chunk_id="pytorch_dl_2",
                    content="To create a DataLoader, you need a Dataset object. The DataLoader then handles batching and shuffling automatically.",
                    score=0.88,
                    metadata={
                        "vendor": "PyTorch",
                        "source_url": "https://pytorch.org/docs/stable/data.html",
                        "heading_path": "PyTorch > DataLoader Tutorial",
                        "doc_type": "tutorial"
                    }
                )
            ],
            "tensorflow dataset": [
                MockRetrievalResult(
                    chunk_id="tf_ds_1",
                    content="TensorFlow's tf.data.Dataset API provides tools to build complex input pipelines from simple, reusable pieces.",
                    score=0.92,
                    metadata={
                        "vendor": "TensorFlow",
                        "source_url": "https://www.tensorflow.org/guide/data",
                        "heading_path": "TensorFlow > Data Pipelines",
                        "doc_type": "documentation"
                    }
                )
            ],
            "mlflow tracking": [
                MockRetrievalResult(
                    chunk_id="mlflow_1",
                    content="MLflow Tracking is a logging API for recording parameters, metrics, and artifacts when running machine learning code.",
                    score=0.90,
                    metadata={
                        "vendor": "MLflow",
                        "source_url": "https://mlflow.org/docs/latest/tracking.html",
                        "heading_path": "MLflow > Tracking",
                        "doc_type": "documentation"
                    }
                )
            ]
        }
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[MockRetrievalResult]:
        """Mock retrieval method."""
        query_lower = query.lower()
        
        # Find best matching mock data
        for key, results in self.mock_data.items():
            if any(term in query_lower for term in key.split()):
                return results[:top_k]
        
        # Return generic result if no match
        return [
            MockRetrievalResult(
                chunk_id="generic_1",
                content=f"Information about {query}...",
                score=0.5,
                metadata={
                    "vendor": "Unknown",
                    "source_url": "https://example.com",
                    "heading_path": "Generic",
                    "doc_type": "documentation"
                }
            )
        ]


class MockTavilySearch:
    """Mock implementation of Tavily web search."""
    
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
    
    def invoke(self, query: str) -> List[Dict[str, Any]]:
        """Mock search method."""
        return [
            {
                "title": f"Result 1 for {query}",
                "url": "https://example.com/result1",
                "content": f"This is a web search result for {query}. It contains relevant information about the topic.",
                "score": 0.95
            },
            {
                "title": f"Result 2 for {query}",
                "url": "https://example.com/result2",
                "content": f"Another result about {query} with additional details and context.",
                "score": 0.88
            }
        ][:self.max_results]


class MockE2BSandbox:
    """Mock implementation of E2B code execution sandbox."""
    
    class MockExecution:
        """Mock execution result."""
        def __init__(self, code: str, should_error: bool = False):
            self.error = None
            self.logs = Mock()
            
            if should_error or "1/0" in code or "raise" in code:
                self.error = "ZeroDivisionError: division by zero"
                self.logs.stdout = []
                self.logs.stderr = [Mock(line="Traceback...\nZeroDivisionError: division by zero\n")]
            else:
                # Simulate successful execution
                if "print" in code:
                    # Extract print content (simplified)
                    import re
                    match = re.search(r'print\(["\'](.+?)["\']\)', code)
                    if match:
                        output = match.group(1) + "\n"
                    else:
                        output = "Execution result\n"
                else:
                    output = "Code executed successfully\n"
                
                self.logs.stdout = [Mock(line=output)]
                self.logs.stderr = []
            
            self.results = None
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def run_code(self, code: str):
        """Mock code execution."""
        # Check for errors in code
        should_error = any(err in code for err in ["1/0", "raise Exception", "undefined_var"])
        return self.MockExecution(code, should_error)


class MockVertexAI:
    """Mock Vertex AI LLM."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    def invoke(self, messages: List) -> Mock:
        """Mock LLM invocation."""
        from langchain_core.messages import AIMessage
        
        # Generate a mock response based on the last message
        last_message = messages[-1] if messages else None
        user_query = last_message.content if last_message else ""
        
        # Simple mock responses based on keywords
        if "pytorch" in user_query.lower() or "dataloader" in user_query.lower():
            response = """PyTorch DataLoader is a powerful utility for loading data in batches. Here's how to use it:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_x, batch_y in dataloader:
    print(batch_x.shape, batch_y.shape)
```

Key features:
- Automatic batching
- Shuffling support
- Multiprocessing for faster data loading

See the official documentation [1] for more details."""
        
        elif "tensorflow" in user_query.lower():
            response = """TensorFlow provides the tf.data.Dataset API for building efficient data pipelines.

Key features:
- Chain transformations
- Prefetching and caching
- Parallel data loading

See TensorFlow documentation [1] for examples."""
        
        elif "mlflow" in user_query.lower():
            response = """MLflow Tracking helps you log and track experiments.

Use mlflow.start_run() to begin tracking and mlflow.log_metric() to log metrics.

Example code:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```"""
        
        else:
            response = f"I can help you with {user_query}. Let me provide some information based on the documentation."
        
        return AIMessage(content=response)
    
    def bind_tools(self, tools: List) -> 'MockVertexAI':
        """Mock tool binding."""
        self.tools = tools
        return self


def create_mock_environment():
    """Create a complete mock environment for testing."""
    return {
        "retriever": MockHybridRetriever(),
        "web_search": MockTavilySearch(),
        "sandbox": MockE2BSandbox(),
        "llm": MockVertexAI()
    }


# Helper functions for patching
def patch_retriever():
    """Context manager to patch the retriever."""
    from unittest.mock import patch
    return patch('agent.tools.retrieval_tools.get_retriever', return_value=MockHybridRetriever())


def patch_tavily():
    """Context manager to patch Tavily search."""
    from unittest.mock import patch
    mock = MockTavilySearch()
    return patch('agent.tools.web_tools.TavilySearchResults', return_value=mock)


def patch_e2b():
    """Context manager to patch E2B sandbox."""
    from unittest.mock import patch
    return patch('agent.tools.code_executor_tool.Sandbox', MockE2BSandbox)


def patch_vertexai():
    """Context manager to patch Vertex AI."""
    from unittest.mock import patch
    return patch('agent.orchestrator.ChatVertexAI', MockVertexAI)


