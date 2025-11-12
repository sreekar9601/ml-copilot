#!/usr/bin/env python3
"""
Testing Environment Setup Script
=================================
Set up the testing environment for ML Docs Copilot.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command with description."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        if e.stderr:
            print(e.stderr)
        return False


def check_python_version():
    """Check Python version."""
    print("\n📋 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ⚠️  Warning: Python 3.10+ recommended")
        return False
    else:
        print("   ✅ Python version OK")
        return True


def check_directory():
    """Check if we're in the right directory."""
    print("\n📋 Checking directory...")
    cwd = Path.cwd()
    
    if (cwd / "tests").exists() or (cwd / "packages" / "api" / "tests").exists():
        print(f"   ✅ Directory OK: {cwd}")
        return True
    else:
        print(f"   ⚠️  Warning: Expected to be in packages/api or project root")
        print(f"   Current: {cwd}")
        return False


def install_dependencies():
    """Install testing dependencies."""
    print("\n📦 Installing testing dependencies...")
    
    # Check if requirements-test.txt exists
    req_file = Path("requirements-test.txt")
    if not req_file.exists():
        print("   ⚠️  requirements-test.txt not found")
        return False
    
    success = run_command(
        f"{sys.executable} -m pip install -r requirements-test.txt",
        "Installing pytest and testing tools"
    )
    
    return success


def install_main_dependencies():
    """Install main application dependencies."""
    print("\n📦 Installing main dependencies...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("   ⚠️  requirements.txt not found")
        return False
    
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing application dependencies"
    )
    
    return success


def check_database():
    """Check if database is set up."""
    print("\n🗄️  Checking database setup...")
    
    data_dir = Path("data")
    qdrant_dir = data_dir / "qdrant"
    
    if qdrant_dir.exists():
        print(f"   ✅ Database directory found: {qdrant_dir}")
        return True
    else:
        print(f"   ⚠️  Database not found: {qdrant_dir}")
        print("   Run ingestion to populate database:")
        print("   python -m ingest.ingest_all")
        return False


def check_env_file():
    """Check environment file."""
    print("\n🔐 Checking environment configuration...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("   ✅ .env file found")
        
        # Check for important variables
        with open(env_file) as f:
            content = f.read()
        
        vars_to_check = [
            "GOOGLE_CLOUD_PROJECT",
            "TAVILY_API_KEY",
            "E2B_API_KEY"
        ]
        
        print("\n   Checking key variables:")
        for var in vars_to_check:
            if var in content and not content.split(var)[1].split('\n')[0].strip() == "=":
                print(f"      ✅ {var}")
            else:
                print(f"      ⚠️  {var} - Not configured")
        
        return True
    else:
        print("   ⚠️  .env file not found")
        print("   Copy env.example to .env and configure:")
        print("   cp env.example .env")
        return False


def run_verification():
    """Run tool verification."""
    print("\n🧪 Running tool verification...")
    
    verification_script = Path("tests/test_tools_verification.py")
    
    if not verification_script.exists():
        print("   ⚠️  Verification script not found")
        return False
    
    success = run_command(
        f"{sys.executable} {verification_script}",
        "Verifying tool configuration",
        check=False  # Don't fail if some tools aren't configured
    )
    
    return True  # Return True even if some tools aren't configured


def create_test_directories():
    """Create necessary test directories."""
    print("\n📁 Creating test directories...")
    
    dirs = [
        "tests/test_data",
        "tests/htmlcov",
        "tests/__pycache__"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
        else:
            print(f"   ✓ Exists: {dir_path}")
    
    return True


def print_summary(checks):
    """Print setup summary."""
    print("\n" + "="*70)
    print("  📊 SETUP SUMMARY")
    print("="*70 + "\n")
    
    total = len(checks)
    passed = sum(1 for check in checks.values() if check)
    
    for name, status in checks.items():
        icon = "✅" if status else "⚠️"
        print(f"  {icon} {name}")
    
    print(f"\n  Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Testing environment is fully set up!")
        print("\nNext steps:")
        print("  1. Run verification: python tests/test_tools_verification.py")
        print("  2. Run unit tests: python run_tests.py --suite unit")
        print("  3. Run all tests: python run_tests.py --coverage")
    else:
        print("\n⚠️  Some components need attention (see above)")
        print("\nYou can still run tests, but some may be skipped:")
        print("  python run_tests.py --suite unit")


def main():
    """Main setup function."""
    print("\n" + "="*70)
    print("  🚀 ML DOCS COPILOT - TESTING SETUP")
    print("="*70)
    
    checks = {}
    
    # Run checks
    checks["Python Version"] = check_python_version()
    checks["Directory"] = check_directory()
    checks["Main Dependencies"] = install_main_dependencies()
    checks["Test Dependencies"] = install_dependencies()
    checks["Test Directories"] = create_test_directories()
    checks["Environment File"] = check_env_file()
    checks["Database"] = check_database()
    checks["Tool Verification"] = run_verification()
    
    # Print summary
    print_summary(checks)
    
    print("\n" + "="*70 + "\n")
    
    # Return success if critical checks passed
    critical_checks = [
        "Python Version",
        "Test Dependencies",
        "Test Directories"
    ]
    
    return all(checks[check] for check in critical_checks if check in checks)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


