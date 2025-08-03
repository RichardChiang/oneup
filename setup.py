"""
Setup script for Chess RL Training System.

Installs dependencies, sets up database, and prepares the system for use.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Backend dependencies
    success, stdout, stderr = run_command("pip install -r backend/requirements.txt")
    if not success:
        print(f"Failed to install backend dependencies: {stderr}")
        return False
    
    # Frontend dependencies  
    success, stdout, stderr = run_command("pip install -r frontend/requirements.txt")
    if not success:
        print(f"Failed to install frontend dependencies: {stderr}")
        return False
    
    print("✓ Dependencies installed successfully")
    return True

def setup_environment():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✓ Environment file created - please review and update .env")
    else:
        print("✓ Environment file already exists")
    
    return True

def test_core_functionality():
    """Test core chess functionality."""
    print("Testing core functionality...")
    
    # Test unicode converter
    success, stdout, stderr = run_command(
        "python -c 'from backend.chess.unicode_converter import test_converter; test_converter()'"
    )
    if not success:
        print(f"Unicode converter test failed: {stderr}")
        return False
    
    # Test chess engine
    success, stdout, stderr = run_command(
        "python -c 'import asyncio; from backend.chess.engine import test_engine; asyncio.run(test_engine())'"
    )
    if not success:
        print(f"Chess engine test failed: {stderr}")
        return False
    
    print("✓ Core functionality tests passed")
    return True

def main():
    """Main setup function."""
    print("🏗️  Setting up Chess RL Training System...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Testing core functionality", test_core_functionality),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update .env file with your settings")
    print("2. Start PostgreSQL database")
    print("3. Run: python scripts/download_lichess_data.py (optional)")
    print("4. Start backend: python -m uvicorn backend.api.main:app --reload")
    print("5. Start frontend: streamlit run frontend/app.py")
    print("\nOr use Docker: docker-compose -f docker/docker-compose.yml up")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)