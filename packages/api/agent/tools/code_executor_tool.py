"""Code execution tool using E2B sandbox."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import os

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Check if E2B is available (check at runtime, not import time)
def _check_e2b_available():
    """Check if E2B is available at runtime."""
    if not os.getenv("E2B_API_KEY"):
        return False, None, "E2B_API_KEY not configured"
    
    # Try different import paths for E2B
    try:
        from e2b_code_interpreter import Sandbox
        return True, Sandbox, None
    except ImportError:
        try:
            from e2b import CodeInterpreter
            # E2B v1.0+ uses CodeInterpreter instead of Sandbox
            return True, CodeInterpreter, None
        except ImportError:
            return False, None, "e2b-code-interpreter package not installed. Install with: pip install e2b-code-interpreter"


class CodeExecutionInput(BaseModel):
    """Input for code execution."""
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Timeout in seconds")


@tool("execute_python_code", args_schema=CodeExecutionInput)
def execute_python_code(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a secure cloud sandbox environment (E2B).
    
    AVAILABLE LIBRARIES (if E2B_TEMPLATE_ID is configured):
    - NumPy, Pandas, Matplotlib - Data analysis and visualization
    - Scikit-learn, SciPy - Machine learning and scientific computing
    - Python standard library
    
    NOT AVAILABLE:
    - PyTorch, TensorFlow (too large for E2B disk limits)
    - For PyTorch/TensorFlow code, explain conceptually instead of executing
    
    Use this tool to:
    - Test NumPy/Pandas data manipulation code
    - Validate algorithms and logic
    - Run calculations with SciPy
    - Test scikit-learn models (small datasets only)
    
    LIMITATIONS:
    - 30-second timeout
    - No GPU access
    - No file system persistence
    - No deep learning frameworks (PyTorch/TensorFlow)
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time (default 30s)
    
    Returns:
        Dictionary with stdout, stderr, execution status, and any errors
    
    Example:
        code = '''
        import numpy as np
        x = np.array([1, 2, 3])
        print(x.shape)
        print(x.mean())
        '''
    """
    
    # Check availability at runtime
    e2b_available, E2BClass, error_msg = _check_e2b_available()
    
    if not e2b_available:
        return {
            "success": False,
            "error": f"Code execution unavailable - {error_msg or 'E2B_API_KEY not configured or e2b-code-interpreter not installed'}",
            "stdout": "",
            "stderr": "",
            "code": code,
            "suggestion": "Install e2b packages: pip install e2b e2b-code-interpreter"
        }
    
    try:
        # Use e2b-code-interpreter Sandbox with create() method
        # Note: Custom templates don't work - code interpreter service not included
        # Using base sandbox (Python 3.11 with standard library only)
        with E2BClass.create(timeout=60) as sandbox:
            # Set execution timeout
            sandbox.set_timeout(timeout)
            # Pass timeout to run_code method
            execution = sandbox.run_code(code, timeout=timeout)
            
            # Extract stdout and stderr from the execution result
            # logs.stdout and logs.stderr are lists of strings
            stdout = ""
            stderr = ""
            
            if hasattr(execution, 'logs') and execution.logs:
                if hasattr(execution.logs, 'stdout') and execution.logs.stdout:
                    stdout = "".join(execution.logs.stdout)
                if hasattr(execution.logs, 'stderr') and execution.logs.stderr:
                    stderr = "".join(execution.logs.stderr)
            
            # Check for errors
            has_error = hasattr(execution, 'error') and execution.error is not None
            error_msg = str(execution.error) if has_error else None
            
            # Get results if available
            results = None
            if hasattr(execution, 'results'):
                results = execution.results
            
            return {
                "success": not has_error,
                "stdout": stdout,
                "stderr": stderr,
                "error": error_msg,
                "code": code,
                "results": results
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Sandbox execution failed: {str(e)}",
            "stdout": "",
            "stderr": "",
            "code": code,
            "suggestion": "Check E2B_API_KEY is valid and e2b package is correctly installed"
        }


class CodeValidationInput(BaseModel):
    """Input for code validation."""
    code: str = Field(description="Python code to validate")


@tool("validate_code_syntax", args_schema=CodeValidationInput)
def validate_code_syntax(code: str) -> dict:
    """
    Check Python code for syntax errors without executing it.
    
    Use this for quick validation before execution.
    
    Args:
        code: Python code to validate
    
    Returns:
        Dictionary with validation status and any syntax errors
    """
    import ast
    
    try:
        ast.parse(code)
        return {
            "valid": True,
            "error": None,
            "code": code
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}",
            "code": code
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}",
            "code": code
        }

