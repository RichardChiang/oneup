"""
Test that all imports work correctly.
"""

import pytest


@pytest.mark.unit
def test_question_service_imports():
    """Test that QuestionService can be imported."""
    try:
        from backend.api.services.question_service import QuestionService
        from backend.api.models import QuestionRequest, QuestionResponse
        assert QuestionService is not None
        assert QuestionRequest is not None
        assert QuestionResponse is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_api_models_imports():
    """Test that API models can be imported."""
    try:
        from backend.api.models import (
            Question, QuestionRequest, QuestionResponse,
            QuestionValidationRequest, QuestionValidationResponse
        )
        assert all([Question, QuestionRequest, QuestionResponse,
                   QuestionValidationRequest, QuestionValidationResponse])
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit 
def test_chess_utils_exist():
    """Test that chess utilities exist."""
    try:
        from backend.chess_utils.engine import ChessEngine
        from backend.chess_utils.unicode_converter import ChessConverter
        assert ChessEngine is not None
        assert ChessConverter is not None
    except ImportError as e:
        pytest.fail(f"Chess utils import failed: {e}")