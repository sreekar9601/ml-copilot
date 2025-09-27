#!/usr/bin/env python3
"""
Convenience script for running ML Documentation Copilot operations.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description=None):
    """Run a command and return success status."""
    if description:
        logger.info(f"📋 {description}")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

async def test_installation():
    """Test the installation."""
    logger.info("🧪 Testing installation...")
    try:
        # Import and run the tests from the test module instead of exec
        import test_installation
        success = await test_installation.run_all_tests()
        return success
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

async def run_ingestion(clear=False, test_mode=False):
    """Run the data ingestion pipeline."""
    logger.info("📥 Running data ingestion...")
    
    cmd = [sys.executable, "-m", "ingest.main"]
    if clear:
        cmd.append("--clear")
    if test_mode:
        cmd.append("--test")
    
    return run_command(cmd, "Ingesting documentation data")

async def run_local_ingestion(clear=False):
    """Run the local documentation ingestion pipeline."""
    logger.info("📥 Running local documentation ingestion...")
    
    cmd = [sys.executable, "-m", "ingest.ingest_local"]
    if clear:
        cmd.append("--clear")
    
    return run_command(cmd, "Ingesting local documentation data")

async def run_focused_ingestion(clear=False):
    """Run the focused documentation ingestion pipeline."""
    logger.info("📥 Running focused documentation ingestion...")
    
    cmd = [sys.executable, "-m", "ingest.ingest_focused"]
    if clear:
        cmd.append("--clear")
    
    return run_command(cmd, "Ingesting focused documentation data")

async def run_comprehensive_ingestion(clear=False):
    """Run the comprehensive documentation ingestion pipeline."""
    logger.info("📥 Running comprehensive documentation ingestion...")
    
    cmd = [sys.executable, "-m", "ingest.ingest_comprehensive"]
    if clear:
        cmd.append("--clear")
    
    return run_command(cmd, "Ingesting comprehensive documentation data")

async def run_pytorch_python_ingestion():
    """Run the PyTorch Python API documentation ingestion pipeline."""
    logger.info("📥 Running PyTorch Python API documentation ingestion...")
    
    cmd = [sys.executable, "-m", "ingest.ingest_pytorch_python"]
    
    return run_command(cmd, "Ingesting PyTorch Python API documentation data")

def start_api(port=8000, reload=False):
    """Start the API server."""
    logger.info(f"🚀 Starting API server on port {port}")
    
    cmd = ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        logger.info("⏹️  API server stopped")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ API server failed: {e}")
        return False

def build_docker():
    """Build Docker image."""
    logger.info("🐳 Building Docker image...")
    
    cmd = ["docker", "build", "-f", "docker/Dockerfile.api", "-t", "ml-docs-copilot", "."]
    return run_command(cmd, "Building Docker image")

def run_docker():
    """Run with Docker Compose."""
    logger.info("🐳 Starting with Docker Compose...")
    
    cmd = ["docker-compose", "up", "--build"]
    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        logger.info("⏹️  Docker services stopped")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Docker run failed: {e}")
        return False

def setup_environment():
    """Setup the development environment."""
    logger.info("⚙️  Setting up development environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        logger.info("📄 Created .env file from env.example")
        logger.warning("⚠️  Please edit .env and add your GOOGLE_API_KEY")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    logger.info(f"📁 Created data directory: {data_dir}")
    
    # Install dependencies
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    if run_command(cmd, "Installing Python dependencies"):
        logger.info("✅ Environment setup complete")
        return True
    else:
        return False

async def full_setup():
    """Run full setup process."""
    logger.info("🔧 Running full setup process...")
    
    steps = [
        ("Setup Environment", setup_environment),
        ("Test Installation", test_installation),
        ("Run Ingestion", lambda: run_ingestion(clear=True)),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Step: {step_name}")
        logger.info('='*50)
        
        try:
            if asyncio.iscoroutinefunction(step_func):
                success = await step_func()
            else:
                success = step_func()
            
            if not success:
                logger.error(f"❌ Step '{step_name}' failed")
                return False
                
            logger.info(f"✅ Step '{step_name}' completed")
            
        except Exception as e:
            logger.error(f"❌ Step '{step_name}' crashed: {e}")
            return False
    
    logger.info("\n" + "="*50)
    logger.info("🎉 Full setup completed successfully!")
    logger.info("="*50)
    logger.info("\nNext steps:")
    logger.info("1. Verify your .env file has GOOGLE_API_KEY set")
    logger.info("2. Run: python run.py start-api")
    logger.info("3. Test at: http://localhost:8000/docs")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="ML Documentation Copilot Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup development environment")
    
    # Full setup command
    full_setup_parser = subparsers.add_parser("full-setup", help="Run complete setup process")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test installation")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run data ingestion")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data")
    ingest_parser.add_argument("--test", action="store_true", help="Test mode with subset")
    
    # Local ingest command
    local_ingest_parser = subparsers.add_parser("ingest-local", help="Run local documentation ingestion")
    local_ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    # Focused ingest command
    focused_ingest_parser = subparsers.add_parser("ingest-focused", help="Run focused documentation ingestion")
    focused_ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    # Comprehensive ingest command
    comprehensive_ingest_parser = subparsers.add_parser("ingest-comprehensive", help="Run comprehensive documentation ingestion")
    comprehensive_ingest_parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    # PyTorch Python API ingest command
    pytorch_ingest_parser = subparsers.add_parser("ingest-pytorch", help="Run PyTorch Python API documentation ingestion")
    
    # API command
    api_parser = subparsers.add_parser("start-api", help="Start API server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Docker commands
    docker_build_parser = subparsers.add_parser("docker-build", help="Build Docker image")
    docker_run_parser = subparsers.add_parser("docker-run", help="Run with Docker Compose")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        success = setup_environment()
        sys.exit(0 if success else 1)
    
    elif args.command == "full-setup":
        success = asyncio.run(full_setup())
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = asyncio.run(test_installation())
        sys.exit(0 if success else 1)
    
    elif args.command == "ingest":
        success = asyncio.run(run_ingestion(args.clear, args.test))
        sys.exit(0 if success else 1)
    
    elif args.command == "ingest-local":
        success = asyncio.run(run_local_ingestion(args.clear))
        sys.exit(0 if success else 1)
    
    elif args.command == "ingest-focused":
        success = asyncio.run(run_focused_ingestion(args.clear))
        sys.exit(0 if success else 1)
    
    elif args.command == "ingest-comprehensive":
        success = asyncio.run(run_comprehensive_ingestion(args.clear))
        sys.exit(0 if success else 1)
    
    elif args.command == "ingest-pytorch":
        success = asyncio.run(run_pytorch_python_ingestion())
        sys.exit(0 if success else 1)
    
    elif args.command == "start-api":
        success = start_api(args.port, args.reload)
        sys.exit(0 if success else 1)
    
    elif args.command == "docker-build":
        success = build_docker()
        sys.exit(0 if success else 1)
    
    elif args.command == "docker-run":
        success = run_docker()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        print("\nCommon workflows:")
        print("  python run.py full-setup     # Complete setup for new installation")
        print("  python run.py start-api      # Start the API server")
        print("  python run.py ingest --clear # Refresh the knowledge base")
        print("  python run.py test           # Test the installation")

if __name__ == "__main__":
    main()
