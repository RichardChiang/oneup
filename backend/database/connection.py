"""
Database connection management for chess RL training system.

Provides async connection pooling, session management, and database utilities.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions for the chess application.
    
    Supports both sync and async operations with proper connection pooling.
    """
    
    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Enable SQL query logging
        """
        self.database_url = database_url
        self.echo = echo
        
        # Parse URL for async version
        if database_url.startswith("postgresql://"):
            self.async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            self.async_database_url = database_url
        
        # Create engines
        self._sync_engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize database engines with proper configuration."""
        try:
            # Sync engine for migrations and admin tasks
            self._sync_engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Async engine for application
            self._async_engine = create_async_engine(
                self.async_database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Attach event listeners to sync engine
            if "sqlite" in self.database_url:
                event.listen(self._sync_engine, "connect", set_sqlite_pragma)
            
            # Session factories
            self._session_factory = sessionmaker(
                bind=self._sync_engine,
                autocommit=False,
                autoflush=False,
            )
            
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
            
            logger.info("Database engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engines: {e}")
            raise
    
    @property
    def sync_engine(self):
        """Get synchronous database engine."""
        return self._sync_engine
    
    @property
    def async_engine(self):
        """Get asynchronous database engine."""
        return self._async_engine
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session."""
        return self._session_factory()
    
    def get_async_session(self) -> AsyncSession:
        """Get asynchronous database session."""
        return self._async_session_factory()
    
    @asynccontextmanager
    async def session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a transactional scope around a series of operations.
        
        Automatically commits on success, rolls back on error.
        """
        session = self.get_async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self._sync_engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self._sync_engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    async def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            async with self.session_scope() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(database_url: Optional[str] = None, echo: bool = False) -> DatabaseManager:
    """
    Initialize global database manager.
    
    Args:
        database_url: Database URL (from env if None)
        echo: Enable SQL logging
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if database_url is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
    
    _db_manager = DatabaseManager(database_url, echo=echo)
    return _db_manager


def get_database() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        DatabaseManager instance
        
    Raises:
        RuntimeError: If database not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_manager


def get_session() -> AsyncSession:
    """
    Get async database session.
    
    Returns:
        AsyncSession instance
    """
    return get_database().get_async_session()


@asynccontextmanager
async def get_session_scope() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with automatic transaction management.
    
    Yields:
        AsyncSession with auto-commit/rollback
    """
    async with get_database().session_scope() as session:
        yield session


# Database event listeners for optimization
# Note: This will be attached to specific engine instances when created
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance (if using SQLite)."""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.close()


async def test_database_connection():
    """Test database connection and basic operations."""
    try:
        # Initialize with test database
        db_url = os.getenv("DATABASE_URL", "postgresql://localhost/chess_rl_test")
        db = initialize_database(db_url, echo=True)
        
        # Test connection
        connected = await db.check_connection()
        print(f"Database connected: {connected}")
        
        # Create tables
        db.create_tables()
        print("Tables created successfully")
        
        # Test session
        async with db.session_scope() as session:
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            print(f"Test query result: {row}")
        
        print("Database test completed successfully!")
        
    except Exception as e:
        print(f"Database test failed: {e}")
        raise
    finally:
        if _db_manager:
            await _db_manager.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_database_connection())