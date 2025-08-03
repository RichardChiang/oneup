"""
Global pytest fixtures and configuration.
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

# Configure async test environment
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_chess_engine():
    """Mock chess engine for testing."""
    engine = Mock()
    engine.evaluate_position = AsyncMock(return_value={
        "score": 0.0,
        "type": "centipawn",
        "best_move": "e4",
        "depth": 15
    })
    engine.get_legal_moves = Mock(return_value=["e4", "d4", "Nf3", "c4"])
    engine.analyze_tactics = AsyncMock(return_value={
        "checks": [],
        "captures": [],
        "threats": []
    })
    return engine


@pytest.fixture
def mock_unicode_converter():
    """Mock unicode converter for testing."""
    converter = Mock()
    converter.fen_to_unicode = Mock(return_value="♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖")
    converter.unicode_to_fen = Mock(return_value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    converter.count_pieces = Mock(return_value={"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1})
    converter.get_piece_at_square = Mock(return_value="K")
    return converter


@pytest.fixture
def mock_database_manager():
    """Mock database manager for testing."""
    db = Mock()
    db.session_scope = AsyncMock()
    return db


@pytest.fixture
def sample_puzzle_data():
    """Sample puzzle data for testing."""
    return {
        "id": "test_puzzle_123",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
        "moves": "e4 e5",
        "rating": 1500,
        "themes": ["opening", "center"],
        "popularity": 85,
        "game_url": "https://lichess.org/abc123"
    }


@pytest.fixture
def starting_position_fen():
    """Standard chess starting position FEN."""
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"