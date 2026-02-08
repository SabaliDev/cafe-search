#!/usr/bin/env python3
"""
Cafe Search API - Local Development Runner

Run the API locally with a single command:
    python run.py

Prerequisites:
    - PostgreSQL 15+ with pgvector extension installed
    - Redis 7+ installed and running
    - Python 3.10+
    - OPENAI_API_KEY environment variable set

The script will:
    1. Check dependencies (PostgreSQL, Redis)
    2. Create virtual environment if needed
    3. Install dependencies
    4. Run database migrations
    5. Start the FastAPI server with auto-reload

For Docker setup (alternative):
    docker-compose up --build
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Configuration
VENV_DIR = Path(".venv")
REQUIRED_PYTHON = (3, 10)
DEFAULT_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/cafesearch"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def check_python_version() -> bool:
    """Check if Python version is sufficient."""
    current = sys.version_info[:2]
    if current < REQUIRED_PYTHON:
        print(f"❌ Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, found {current[0]}.{current[1]}")
        return False
    print(f"✅ Python {current[0]}.{current[1]} found")
    return True


def check_postgres() -> bool:
    """Check if PostgreSQL is running and pgvector is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            print(f"✅ PostgreSQL client found (v{version})")
            
            # Check if pgvector extension is available
            check_cmd = [
                "psql",
                "-d", "postgres",
                "-c", "SELECT * FROM pg_available_extensions WHERE name = 'vector';"
            ]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if "vector" in result.stdout:
                print("✅ pgvector extension available")
                return True
            else:
                print("⚠️  pgvector extension not found. Install with: brew install pgvector (macOS) or apt install postgresql-15-pgvector (Ubuntu)")
                return False
        return False
    except FileNotFoundError:
        print("❌ PostgreSQL client not found. Install PostgreSQL 15+ first.")
        print("   macOS: brew install postgresql@15")
        print("   Ubuntu: sudo apt install postgresql-15")
        return False
    except Exception as e:
        print(f"⚠️  Could not verify PostgreSQL: {e}")
        return True  # Continue anyway, might work


def check_redis() -> bool:
    """Check if Redis is running."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 6379))
        sock.close()
        
        if result == 0:
            print("✅ Redis running on localhost:6379")
            return True
        else:
            print("⚠️  Redis not running on localhost:6379")
            print("   Start with: redis-server")
            return False
    except Exception as e:
        print(f"⚠️  Could not verify Redis: {e}")
        return False


def setup_virtual_environment() -> Path:
    """Create and return virtual environment path."""
    if not VENV_DIR.exists():
        print(f"📦 Creating virtual environment in {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    
    # Determine pip path
    if os.name == "nt":  # Windows
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:  # Unix/Mac
        pip_path = VENV_DIR / "bin" / "pip"
        python_path = VENV_DIR / "bin" / "python"
    
    return python_path, pip_path


def install_dependencies(pip_path: Path) -> bool:
    """Install required packages."""
    print("📥 Installing dependencies...")
    
    requirements_files = ["requirements.txt", "requirements-dev.txt"]
    for req_file in requirements_files:
        if Path(req_file).exists():
            result = subprocess.run([str(pip_path), "install", "-r", req_file])
            if result.returncode != 0:
                print(f"❌ Failed to install {req_file}")
                return False
    
    print("✅ Dependencies installed")
    return True


def check_env_variables() -> bool:
    """Check and set environment variables."""
    # Load from .env file if exists
    env_file = Path(".env")
    if env_file.exists():
        print("📄 Loading environment from .env")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value.strip().strip('"').strip("'")
    
    # Set defaults if not already set
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = DEFAULT_DB_URL
        print(f"📍 Using default DATABASE_URL: {DEFAULT_DB_URL}")
    else:
        print(f"📍 DATABASE_URL: {os.environ['DATABASE_URL']}")
    
    if "REDIS_URL" not in os.environ:
        os.environ["REDIS_URL"] = DEFAULT_REDIS_URL
        print(f"📍 Using default REDIS_URL: {DEFAULT_REDIS_URL}")
    else:
        print(f"📍 REDIS_URL: {os.environ['REDIS_URL']}")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  OPENAI_API_KEY not set. Some features will fail.")
        print("   Set with: export OPENAI_API_KEY='your-key'")
    else:
        masked = os.environ["OPENAI_API_KEY"][:8] + "..."
        print(f"📍 OPENAI_API_KEY: {masked}")
    
    return True


def run_migrations(python_path: Path) -> bool:
    """Run Alembic migrations."""
    print("🔄 Running database migrations...")
    result = subprocess.run([str(python_path), "-m", "alembic", "upgrade", "head"])
    if result.returncode != 0:
        print("❌ Migration failed. Is PostgreSQL running?")
        print("   Start PostgreSQL: pg_ctl -D /usr/local/var/postgresql@15 start")
        return False
    print("✅ Migrations complete")
    return True


def start_server(python_path: Path) -> None:
    """Start the FastAPI development server."""
    print("\n" + "=" * 60)
    print("🚀 Starting Cafe Search API...")
    print("=" * 60)
    print(f"\nAPI will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            str(python_path), "-m", "uvicorn",
            "app.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Cafe Search API - Local Development Setup")
    print("=" * 60 + "\n")
    
    # Pre-flight checks
    if not check_python_version():
        return 1
    
    # Check services (warn but don't fail)
    check_postgres()
    check_redis()
    
    # Setup environment
    print(f"\n📦 Setting up virtual environment...")
    python_path, pip_path = setup_virtual_environment()
    
    if not install_dependencies(pip_path):
        return 1
    
    if not check_env_variables():
        return 1
    
    # Run migrations
    if not run_migrations(python_path):
        return 1
    
    # Start server
    start_server(python_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
