"""
Unit tests for QuestionService using TDD approach.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import chess

from backend.api.services.question_service import QuestionService
from backend.api.models import QuestionRequest, Question


@pytest.mark.unit
class TestQuestionService:
    """Test suite for QuestionService."""
    
    @pytest.fixture
    def question_service(self, mock_database_manager, mock_chess_engine, mock_unicode_converter):
        """Create QuestionService instance with mocked dependencies."""
        return QuestionService(mock_database_manager, mock_chess_engine, mock_unicode_converter)
    
    @pytest.mark.asyncio
    async def test_generate_questions_success(self, question_service):
        """Test successful question generation."""
        # Arrange
        request = QuestionRequest(level=1, count=3)
        
        # Mock database response
        with patch.object(question_service, '_get_position_data', return_value={
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
            "board": chess.Board(),
            "themes": ["opening"],
            "rating": 1200,
            "puzzle_id": "test123"
        }):
            # Act
            response = await question_service.generate_questions(request)
        
        # Assert
        assert response.total_generated == 3
        assert response.level == 1
        assert len(response.questions) == 3
        assert all(q.level == 1 for q in response.questions)
        assert response.generation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_specific_fen(self, question_service):
        """Test question generation with specific FEN position."""
        # Arrange
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        request = QuestionRequest(level=2, count=1, fen=test_fen)
        
        # Act
        response = await question_service.generate_questions(request)
        
        # Assert
        assert response.total_generated == 1
        assert response.questions[0].fen == test_fen
    
    @pytest.mark.asyncio
    async def test_generate_questions_empty_response(self, question_service):
        """Test handling of empty question generation."""
        # Arrange
        request = QuestionRequest(level=1, count=5)
        
        # Mock empty position data
        with patch.object(question_service, '_get_position_data', return_value=None):
            # Act
            response = await question_service.generate_questions(request)
        
        # Assert
        assert response.total_generated == 0
        assert len(response.questions) == 0
    
    @pytest.mark.asyncio
    async def test_validate_question_valid(self, question_service):
        """Test validation of a valid question."""
        # Arrange
        question = Question(
            id="q_test_001",
            level=1,
            question_type="piece_count",
            question_text="How many pawns does White have?",
            correct_answer="8",
            alternative_answers=[],
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            unicode_position="♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
            themes=["basics"]
        )
        
        # Act
        result = await question_service.validate_question(question)
        
        # Assert
        assert result.is_valid
        assert result.chess_validity
        assert result.answer_correctness
        assert result.difficulty_match
        assert result.validation_score >= 0.7
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_question_invalid_fen(self, question_service):
        """Test validation with invalid FEN."""
        # Arrange
        question = Question(
            id="q_test_002",
            level=1,
            question_type="piece_count",
            question_text="How many pieces?",
            correct_answer="16",
            alternative_answers=[],
            fen="invalid_fen_notation",
            unicode_position="invalid",
            themes=[]
        )
        
        # Act
        result = await question_service.validate_question(question)
        
        # Assert
        assert not result.is_valid
        assert not result.chess_validity
        assert "Invalid chess position" in result.errors


@pytest.mark.unit
class TestQuestionGenerationByLevel:
    """Test question generation for each difficulty level."""
    
    @pytest.fixture
    def question_service(self, mock_database_manager, mock_chess_engine, mock_unicode_converter):
        """Create QuestionService instance."""
        return QuestionService(mock_database_manager, mock_chess_engine, mock_unicode_converter)
    
    @pytest.fixture
    def position_data(self):
        """Standard position data for testing."""
        return {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
            "board": chess.Board(),
            "themes": ["opening"],
            "rating": 1500,
            "puzzle_id": None
        }
    
    @pytest.mark.asyncio
    async def test_level1_piece_counting(self, question_service, position_data):
        """Test Level 1 piece counting questions."""
        # Act
        question_data = await question_service._generate_level1_question(
            "piece_count", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "How many" in question_data["question"]
        assert question_data["answer"].isdigit()
        assert "explanation" in question_data
    
    @pytest.mark.asyncio
    async def test_level1_material_count(self, question_service, position_data):
        """Test Level 1 material counting questions."""
        # Act
        question_data = await question_service._generate_level1_question(
            "material_count", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "Which side has more pieces" in question_data["question"]
        assert question_data["answer"] in ["White", "Black", "Equal"]
    
    @pytest.mark.asyncio
    async def test_level2_piece_position(self, question_service, position_data):
        """Test Level 2 piece position questions."""
        # Act
        question_data = await question_service._generate_level2_question(
            "piece_position", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "What piece is on" in question_data["question"]
        assert any(piece in question_data["answer"] for piece in 
                  ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"])
    
    @pytest.mark.asyncio
    async def test_level2_square_identification(self, question_service, position_data):
        """Test Level 2 square identification questions."""
        # Act
        question_data = await question_service._generate_level2_question(
            "square_identification", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "which square" in question_data["question"].lower()
        # Answer should be a valid square notation
        assert len(question_data["answer"]) == 2
        assert question_data["answer"][0] in "abcdefgh"
        assert question_data["answer"][1] in "12345678"
    
    @pytest.mark.asyncio
    async def test_level3_checks(self, question_service, position_data):
        """Test Level 3 check detection questions."""
        # Act
        question_data = await question_service._generate_level3_question(
            "checks", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "check" in question_data["question"].lower()
        assert question_data["answer"] in ["Yes", "No"]
    
    @pytest.mark.asyncio
    async def test_level4_pawn_structure(self, question_service, position_data):
        """Test Level 4 pawn structure questions."""
        # Act
        question_data = await question_service._generate_level4_question(
            "pawn_structure", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        assert "pawn structure" in question_data["question"].lower()
        assert "pawn" in question_data["explanation"].lower()
    
    @pytest.mark.asyncio
    async def test_level5_best_moves(self, question_service, position_data, mock_chess_engine):
        """Test Level 5 best move questions."""
        # Act
        question_data = await question_service._generate_level5_question(
            "best_moves", position_data["board"], position_data
        )
        
        # Assert
        assert question_data is not None
        # Should either get best move or fallback question
        assert ("best move" in question_data["question"].lower() or 
                "capture" in question_data["question"].lower())


@pytest.mark.unit
class TestQuestionTypes:
    """Test specific question type generation."""
    
    @pytest.fixture
    def question_service(self, mock_database_manager, mock_chess_engine, mock_unicode_converter):
        """Create QuestionService instance."""
        return QuestionService(mock_database_manager, mock_chess_engine, mock_unicode_converter)
    
    def test_level_definitions(self, question_service):
        """Test that all levels are properly defined."""
        # Assert
        assert len(question_service.level_definitions) == 5
        
        for level in range(1, 6):
            assert level in question_service.level_definitions
            level_def = question_service.level_definitions[level]
            assert "name" in level_def
            assert "types" in level_def
            assert "description" in level_def
            assert len(level_def["types"]) >= 3
    
    @pytest.mark.asyncio
    async def test_get_position_data_from_fen(self, question_service, mock_unicode_converter):
        """Test getting position data from FEN."""
        # Arrange
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Act
        position_data = await question_service._get_position_data(test_fen, None, 1)
        
        # Assert
        assert position_data is not None
        assert position_data["fen"] == test_fen
        assert "board" in position_data
        assert isinstance(position_data["board"], chess.Board)
        mock_unicode_converter.fen_to_unicode.assert_called_once_with(test_fen)
    
    def test_validate_chess_position_valid(self, question_service):
        """Test chess position validation with valid FEN."""
        # Arrange
        valid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Act
        is_valid = question_service._validate_chess_position(valid_fen)
        
        # Assert
        assert is_valid
    
    def test_validate_chess_position_invalid(self, question_service):
        """Test chess position validation with invalid FEN."""
        # Arrange
        invalid_fen = "invalid_fen_string"
        
        # Act
        is_valid = question_service._validate_chess_position(invalid_fen)
        
        # Assert
        assert not is_valid
    
    def test_validate_difficulty_level_match(self, question_service):
        """Test difficulty level validation."""
        # Arrange
        question = Question(
            id="test",
            level=1,
            question_type="piece_count",  # Valid for level 1
            question_text="Test",
            correct_answer="Test",
            fen="test",
            unicode_position="test"
        )
        
        # Act
        is_valid = question_service._validate_difficulty_level(question)
        
        # Assert
        assert is_valid
    
    def test_validate_difficulty_level_mismatch(self, question_service):
        """Test difficulty level validation with mismatch."""
        # Arrange
        question = Question(
            id="test",
            level=1,
            question_type="complex_tactics",  # Not valid for level 1
            question_text="Test",
            correct_answer="Test",
            fen="test",
            unicode_position="test"
        )
        
        # Act
        is_valid = question_service._validate_difficulty_level(question)
        
        # Assert
        assert not is_valid
    
    def test_calculate_validation_score(self, question_service):
        """Test validation score calculation."""
        # Test perfect score
        score = question_service._calculate_validation_score(
            chess_validity=True,
            answer_correctness=True,
            difficulty_match=True,
            error_count=0,
            warning_count=0
        )
        assert score == 1.0
        
        # Test with errors
        score = question_service._calculate_validation_score(
            chess_validity=False,
            answer_correctness=True,
            difficulty_match=True,
            error_count=2,
            warning_count=1
        )
        assert score < 0.7
        
        # Test minimum score
        score = question_service._calculate_validation_score(
            chess_validity=False,
            answer_correctness=False,
            difficulty_match=False,
            error_count=5,
            warning_count=5
        )
        assert score >= 0.05  # Minimum score guaranteed