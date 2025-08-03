"""
Simple unit tests for QuestionService core functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import chess

from backend.api.services.question_service import QuestionService
from backend.api.models import QuestionRequest


class TestQuestionServiceBasic:
    """Basic tests for QuestionService."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database manager."""
        return Mock()
    
    @pytest.fixture
    def mock_engine(self):
        """Mock chess engine."""
        engine = Mock()
        engine.evaluate_position = AsyncMock(return_value={"score": 0.0, "best_move": "e4"})
        engine.analyze_tactics = AsyncMock(return_value={"checks": [], "captures": []})
        return engine
    
    @pytest.fixture
    def mock_converter(self):
        """Mock chess converter."""
        converter = Mock()
        converter.fen_to_unicode = Mock(return_value="unicode_board")
        return converter
    
    @pytest.fixture
    def service(self, mock_db, mock_engine, mock_converter):
        """Create service instance."""
        return QuestionService(mock_db, mock_engine, mock_converter)
    
    def test_init(self, service):
        """Test service initialization."""
        assert service.db is not None
        assert service.engine is not None
        assert service.converter is not None
        assert len(service.level_definitions) == 5
    
    def test_level_definitions(self, service):
        """Test level definitions are complete."""
        for level in range(1, 6):
            assert level in service.level_definitions
            level_def = service.level_definitions[level]
            assert "name" in level_def
            assert "types" in level_def
            assert "description" in level_def
    
    def test_validate_chess_position_valid(self, service):
        """Test chess position validation."""
        valid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert service._validate_chess_position(valid_fen) is True
    
    def test_validate_chess_position_invalid(self, service):
        """Test invalid chess position validation."""
        invalid_fen = "invalid_fen"
        assert service._validate_chess_position(invalid_fen) is False
    
    @pytest.mark.asyncio
    async def test_get_position_data_from_fen(self, service):
        """Test getting position data from FEN."""
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        position_data = await service._get_position_data(test_fen, None, 1)
        
        assert position_data is not None
        assert position_data["fen"] == test_fen
        assert isinstance(position_data["board"], chess.Board)
    
    @pytest.mark.asyncio
    async def test_level1_piece_counting(self, service):
        """Test Level 1 piece counting question generation."""
        board = chess.Board()
        position_data = {"fen": board.fen(), "board": board}
        
        question_data = await service._generate_level1_question(
            "piece_count", board, position_data
        )
        
        assert question_data is not None
        assert "question" in question_data
        assert "answer" in question_data
        assert "How many" in question_data["question"]
    
    @pytest.mark.asyncio
    async def test_level2_piece_position(self, service):
        """Test Level 2 piece position question generation."""
        board = chess.Board()
        position_data = {"fen": board.fen(), "board": board}
        
        question_data = await service._generate_level2_question(
            "piece_position", board, position_data
        )
        
        assert question_data is not None
        assert "What piece is on" in question_data["question"]