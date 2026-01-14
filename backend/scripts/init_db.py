"""Initialize database with tables."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings, ensure_directories
from database.connection import init_db
from loguru import logger


def main():
    """Initialize database."""
    logger.info("Initializing database...")
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    
    # Ensure directories exist
    ensure_directories()
    logger.info("Directories created")
    
    # Create tables
    init_db()
    logger.info("Database initialization complete!")


if __name__ == "__main__":
    main()
