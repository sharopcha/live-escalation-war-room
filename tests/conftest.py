"""
Pytest configuration — adds project root to sys.path and loads .env
so that OPENAI_API_KEY is available to the embedding retriever tests.
"""
import sys
from pathlib import Path

# Make shared/, agents/, bridge/ importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if python-dotenv is available (it is via pydantic-settings)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
