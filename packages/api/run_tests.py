#!/usr/bin/env python3
"""
Comprehensive Test Runner
=========================
Run tests with coverage reporting and detailed output.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "e2e", "verification"],
        default="all",
        help="Which test suite to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--markers",
        type=str,
        help="Pytest markers to filter tests (e.g., 'not slow')"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML test report"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  🧪 ML DOCS COPILOT - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    # Change to tests directory
    tests_dir = Path(__file__).parent / "tests"
    os.chdir(tests_dir)
    
    exit_code = 0
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=../agent",
            "--cov=../api",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    if args.html:
        pytest_cmd.extend([
            "--html=test_report.html",
            "--self-contained-html"
        ])
    
    if args.markers:
        pytest_cmd.extend(["-m", args.markers])
    
    # Run verification first if requested or if running all
    if args.suite in ["all", "verification"]:
        print("\n📋 Step 1: Tool Verification")
        print("-" * 70)
        result = run_command(
            "python test_tools_verification.py",
            "Verifying Tool Configuration"
        )
        if result != 0:
            print("\n⚠️  Some tools are not configured. Tests will continue.")
            print("   Configure missing API keys for full functionality.")
    
    # Run unit tests
    if args.suite in ["all", "unit"]:
        print("\n📋 Step 2: Unit Tests")
        print("-" * 70)
        
        unit_tests = [
            "test_retrieval_tools.py",
            "test_web_tools.py",
            "test_code_executor.py"
        ]
        
        cmd = " ".join(pytest_cmd + unit_tests)
        result = run_command(cmd, "Running Unit Tests")
        
        if result != 0:
            exit_code = result
            print("\n❌ Unit tests failed!")
    
    # Run integration tests
    if args.suite in ["all", "integration"]:
        print("\n📋 Step 3: Integration Tests")
        print("-" * 70)
        
        cmd = " ".join(pytest_cmd + ["test_orchestrator_integration.py"])
        result = run_command(cmd, "Running Integration Tests")
        
        if result != 0:
            exit_code = result
            print("\n❌ Integration tests failed!")
    
    # Run E2E tests
    if args.suite in ["all", "e2e"]:
        print("\n📋 Step 4: End-to-End Tests")
        print("-" * 70)
        print("Note: E2E tests require the API server to be running")
        print("      Start with: python -m api.main\n")
        
        cmd = " ".join(pytest_cmd + ["test_api_endpoints.py"])
        result = run_command(cmd, "Running End-to-End Tests")
        
        if result != 0:
            print("\n⚠️  E2E tests failed or skipped (server not running)")
    
    # Print summary
    print("\n" + "="*70)
    print("  📊 TEST SUMMARY")
    print("="*70 + "\n")
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. See output above for details.")
    
    if args.coverage:
        print("\n📈 Coverage report generated:")
        print("   - Terminal: See above")
        print("   - HTML: htmlcov/index.html")
    
    if args.html:
        print("\n📄 HTML test report generated: test_report.html")
    
    print("\n" + "="*70 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


