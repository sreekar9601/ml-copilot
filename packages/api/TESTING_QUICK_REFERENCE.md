# Testing Quick Reference

## 🚀 One-Command Setup

```bash
cd packages/api
python setup_testing.py
```

## ⚡ Common Commands

### Verify Tools
```bash
python tests/test_tools_verification.py
```

### Run All Tests
```bash
python run_tests.py --coverage
```

### Run Specific Suite
```bash
python run_tests.py --suite unit        # Unit tests only
python run_tests.py --suite integration # Integration tests
python run_tests.py --suite e2e         # End-to-end tests (requires server)
```

### Run With Coverage
```bash
python run_tests.py --coverage --html
# View: tests/htmlcov/index.html
```

### Run Specific Test File
```bash
pytest tests/test_retrieval_tools.py -v
```

### Run Specific Test
```bash
pytest tests/test_retrieval_tools.py::TestHybridDocSearch::test_basic_search -v
```

## 🔍 Tool Status Check

```bash
python tests/test_tools_verification.py
```

**Output**:
- ✅ Working tools
- ⚠️  Not configured (need API keys)
- ❌ Broken/Error

## 📊 Test Results

### Understanding Output

```
✅ PASS - Test passed completely
❌ FAIL - Test failed
⚠️  SKIP - Test skipped (missing dependencies)
```

### Coverage Report

```
----------- coverage -----------
Name                          Stmts   Miss  Cover
-------------------------------------------------
agent/tools/retrieval.py        150     10    93%
agent/orchestrator.py           250     20    92%
-------------------------------------------------
TOTAL                           400     30    92%
```

## 🎯 What to Test

| Component | Test File | Command |
|-----------|-----------|---------|
| Retrieval | `test_retrieval_tools.py` | `pytest tests/test_retrieval_tools.py` |
| Web Search | `test_web_tools.py` | `pytest tests/test_web_tools.py` |
| Code Exec | `test_code_executor.py` | `pytest tests/test_code_executor.py` |
| Orchestrator | `test_orchestrator_integration.py` | `pytest tests/test_orchestrator_integration.py` |
| API | `test_api_endpoints.py` | Start server + `pytest tests/test_api_endpoints.py` |

## 🔐 API Keys

### Required for Full Testing

```bash
# .env file
TAVILY_API_KEY=your_key        # For web search
E2B_API_KEY=your_key           # For code execution
GOOGLE_CLOUD_PROJECT=project   # For LLM
```

### Testing Without Keys

Most tests work without API keys - they automatically use mocks or skip.

```bash
# Run tests that don't need keys
pytest -m "not requires_api_keys"
```

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### "Connection refused" (E2E tests)
```bash
# Terminal 1: Start server
python -m api.main

# Terminal 2: Run tests
pytest tests/test_api_endpoints.py
```

### "No coverage"
```bash
pip install pytest-cov
```

### Database not found
```bash
python -m ingest.ingest_all
```

## 📈 CI/CD Integration

### GitHub Actions
```yaml
- name: Run tests
  run: |
    cd packages/api
    python run_tests.py --coverage
```

### Pre-commit Hook
```bash
#!/bin/bash
cd packages/api
python tests/test_tools_verification.py
python run_tests.py --suite unit
```

## 🎓 Best Practices

1. **Run verification first**: `python tests/test_tools_verification.py`
2. **Test locally before commit**: `python run_tests.py --suite unit`
3. **Check coverage**: `python run_tests.py --coverage`
4. **Test one component**: `pytest tests/test_retrieval_tools.py -v`
5. **Use markers**: `pytest -m "not slow"`

## 📚 Full Documentation

See `tests/README.md` for comprehensive testing guide.

## 🆘 Quick Help

```bash
pytest --help                    # Pytest help
python run_tests.py --help       # Test runner help
pytest --markers                 # List available markers
pytest --fixtures                # List available fixtures
```

## ⏱️ Performance Benchmarks

| Test Suite | Expected Time |
|------------|---------------|
| Verification | < 10 seconds |
| Unit Tests | < 30 seconds |
| Integration | < 2 minutes |
| E2E Tests | < 5 minutes |
| Full Suite | < 5 minutes |

## 🎯 Success Criteria

- ✅ All unit tests pass
- ✅ Coverage > 80%
- ✅ No critical tools broken
- ✅ Integration tests pass
- ✅ E2E tests pass (if server running)

## 🔄 Continuous Testing

### Watch Mode (Optional)
```bash
pip install pytest-watch
ptw tests/ -- -v
```

### Parallel Execution
```bash
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
```

## 📞 Support

Issues? Check:
1. Environment variables configured?
2. Dependencies installed?
3. Database populated?
4. Server running (for E2E)?
5. Correct directory (`packages/api`)?

Full docs: `tests/README.md`


