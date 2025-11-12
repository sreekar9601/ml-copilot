# ML Docs Copilot - Testing Guide

## 📋 Overview

This directory contains a comprehensive test suite for the ML Docs Copilot system, including:

- **Tool Verification** - Check which tools are configured and functional
- **Unit Tests** - Test individual tools and components
- **Integration Tests** - Test the agent orchestrator and workflows
- **End-to-End Tests** - Test the complete API
- **Mock Implementations** - Mock external dependencies for testing

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
cd packages/api
pip install -r requirements-test.txt
```

### 2. Run Tool Verification

Check which tools are configured and working:

```bash
python tests/test_tools_verification.py
```

This will show you:
- ✅ Which tools are working
- ⚠️  Which tools need API key configuration
- ❌ Which tools have errors

### 3. Run All Tests

```bash
python run_tests.py --suite all --coverage
```

## 📁 Test Structure

```
tests/
├── __init__.py                          # Package initialization
├── conftest.py                          # Pytest fixtures and configuration
├── mocks.py                             # Mock implementations
├── README.md                            # This file
│
├── test_tools_verification.py           # Tool verification script
├── test_retrieval_tools.py              # Unit tests for retrieval
├── test_web_tools.py                    # Unit tests for web search
├── test_code_executor.py                # Unit tests for code execution
├── test_orchestrator_integration.py     # Integration tests
└── test_api_endpoints.py                # End-to-end API tests
```

## 🧪 Test Suites

### Tool Verification

**Purpose**: Check which tools are configured and functional

**Run**:
```bash
python tests/test_tools_verification.py
```

**Checks**:
- Environment variables (API keys)
- Retrieval tools (vector search, BM25)
- Web search (Tavily)
- Code execution (E2B)
- Code validation (AST parsing)

**Output**:
```
✅ hybrid_doc_search              working
✅ get_specific_documentation     working
⚠️  web_search                    not_configured
⚠️  execute_python_code           not_configured
✅ validate_code_syntax           working
```

### Unit Tests

**Purpose**: Test individual tools and components in isolation

**Run**:
```bash
python run_tests.py --suite unit
```

**Coverage**:
- `test_retrieval_tools.py` - Hybrid search, specific documentation lookup
- `test_web_tools.py` - Web search functionality and error handling
- `test_code_executor.py` - Code execution and syntax validation

**Example**:
```bash
# Run specific test file
pytest tests/test_retrieval_tools.py -v

# Run specific test
pytest tests/test_retrieval_tools.py::TestHybridDocSearch::test_basic_search -v

# Run with markers
pytest -m "not slow" -v
```

### Integration Tests

**Purpose**: Test the agent orchestrator and complete workflows

**Run**:
```bash
python run_tests.py --suite integration
```

**Coverage**:
- Agent orchestrator functionality
- Tool calling and routing
- Multi-turn conversations
- Cost tracking
- Iteration limits

**Example**:
```bash
pytest tests/test_orchestrator_integration.py -v
```

### End-to-End Tests

**Purpose**: Test the complete API with HTTP requests

**Requirements**: API server must be running

**Start Server**:
```bash
# Terminal 1: Start server
python -m api.main

# Terminal 2: Run E2E tests
python run_tests.py --suite e2e
```

**Coverage**:
- HTTP endpoints
- Request/response formats
- Error handling
- Streaming responses
- Concurrent requests

## 🔧 Test Runner Options

The `run_tests.py` script provides comprehensive test running with various options:

### Basic Usage

```bash
# Run all tests
python run_tests.py

# Run specific suite
python run_tests.py --suite unit
python run_tests.py --suite integration
python run_tests.py --suite e2e
python run_tests.py --suite verification

# Run with coverage
python run_tests.py --coverage

# Run with HTML report
python run_tests.py --html

# Verbose output
python run_tests.py --verbose

# Filter by markers
python run_tests.py --markers "not slow"
```

### Advanced Usage

```bash
# Run unit tests with coverage and HTML report
python run_tests.py --suite unit --coverage --html --verbose

# Run fast tests only
python run_tests.py --markers "not slow" --verbose

# Run integration tests without coverage
python run_tests.py --suite integration -v
```

## 🎭 Mocking External Dependencies

When API keys are not available, tests can use mocks:

### Available Mocks

```python
from tests.mocks import (
    MockHybridRetriever,
    MockTavilySearch,
    MockE2BSandbox,
    MockVertexAI,
    create_mock_environment
)

# Use in tests
def test_with_mock():
    retriever = MockHybridRetriever()
    results = retriever.retrieve("PyTorch DataLoader")
    assert len(results) > 0
```

### Fixtures

Pytest fixtures are available in `conftest.py`:

```python
def test_with_fixtures(mock_tavily_search, mock_e2b_sandbox):
    # Mocks are automatically applied
    result = web_search("PyTorch")
    assert result is not None
```

## 📊 Coverage Reports

### Generate Coverage

```bash
python run_tests.py --coverage
```

### View Reports

**Terminal Report**: Shows immediately after tests
```
----------- coverage: platform win32, python 3.12 -----------
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
agent/tools/retrieval_tools.py         150     10    93%   45-50
agent/tools/web_tools.py                85      5    94%   22-25
agent/orchestrator.py                  250     20    92%   180-195
------------------------------------------------------------------
TOTAL                                  485     35    93%
```

**HTML Report**: Open `tests/htmlcov/index.html` in browser
- Interactive coverage exploration
- Line-by-line highlighting
- Branch coverage details

## 🔍 Test Scenarios

### Golden Test Set

The evaluation module includes comprehensive test scenarios in `agent/evaluation/test_scenarios.py`:

**Categories**:
- Basic documentation queries
- Multi-framework comparisons
- Debugging scenarios
- Code execution
- Tutorials/How-tos
- Web search required
- Conceptual questions
- API references
- Best practices
- Edge cases

**Run Evaluation**:
```bash
python -m agent.evaluation.metrics
```

## ⚙️ Configuration

### Environment Variables for Testing

Create a `.env.test` file:

```bash
# Required for full functionality
TAVILY_API_KEY=your_tavily_key_here
E2B_API_KEY=your_e2b_key_here

# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp-credentials.json

# Optional
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=false
```

### Pytest Configuration

Create `pytest.ini` in the `packages/api` directory:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -ra
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    requires_api_keys: marks tests requiring API keys
asyncio_mode = auto
```

## 🐛 Debugging Tests

### Run Single Test with Debug Output

```bash
pytest tests/test_retrieval_tools.py::TestHybridDocSearch::test_basic_search -v -s
```

### Use Pytest Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace
```

### Print Debug Information

```python
def test_with_debug():
    result = hybrid_doc_search("PyTorch")
    print(f"DEBUG: Result = {result}")  # Will show with -s flag
    assert result is not None
```

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          cd packages/api
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run verification
        run: python packages/api/tests/test_tools_verification.py
      
      - name: Run unit tests
        run: python packages/api/run_tests.py --suite unit --coverage
      
      - name: Run integration tests
        run: python packages/api/run_tests.py --suite integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 🔐 Testing Without API Keys

Many tests work without external API keys:

### What Works Without Keys

- ✅ Retrieval tools (if database is set up)
- ✅ Code syntax validation
- ✅ Mock-based tests
- ✅ Structure validation tests

### What Requires Keys

- ⚠️  Web search (needs TAVILY_API_KEY)
- ⚠️  Code execution (needs E2B_API_KEY)
- ⚠️  LLM calls (needs GCP credentials)

### Running Without Keys

```bash
# Tests will automatically skip or use mocks
python run_tests.py --suite unit

# Explicitly use only mock tests
pytest -m "not requires_api_keys"
```

## 📋 Best Practices

### Writing New Tests

1. **Use descriptive names**: `test_search_returns_relevant_results`
2. **Test one thing**: Each test should verify one behavior
3. **Use fixtures**: Reuse common setup with fixtures
4. **Add markers**: Mark slow/integration/e2e tests appropriately
5. **Mock external calls**: Use mocks for external APIs
6. **Assert meaningfully**: Check structure and content

### Example Test

```python
import pytest

class TestMyFeature:
    """Test my new feature."""
    
    def test_basic_functionality(self, sample_query):
        """Test that basic functionality works."""
        result = my_function(sample_query)
        
        assert isinstance(result, dict)
        assert "key" in result
        assert len(result["key"]) > 0
    
    @pytest.mark.slow
    @pytest.mark.requires_api_keys
    def test_with_external_api(self):
        """Test with real API call."""
        result = call_external_api()
        assert result.status_code == 200
```

## 🆘 Troubleshooting

### Tests Fail with "Module not found"

```bash
# Make sure you're in the right directory
cd packages/api

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Tests Fail with "Connection refused"

E2E tests need the server running:
```bash
# Terminal 1
python -m api.main

# Terminal 2
pytest tests/test_api_endpoints.py
```

### Tests Fail with "API key not configured"

Either:
1. Configure the API key in `.env`
2. Run tests that don't require it: `pytest -m "not requires_api_keys"`
3. Tests will automatically skip or use mocks

### Coverage Report Not Generating

```bash
# Install coverage package
pip install pytest-cov

# Run with coverage flag
python run_tests.py --coverage
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contributing

When adding new features:

1. ✅ Write tests first (TDD)
2. ✅ Ensure >80% coverage
3. ✅ Run verification before committing
4. ✅ Update test scenarios if needed
5. ✅ Document new test fixtures

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review test output carefully
- Check environment configuration
- Verify database setup for retrieval tests


