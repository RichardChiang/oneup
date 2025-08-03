"""
Pydantic models for API request/response validation.

Defines data structures for chat, feedback, puzzles, and health checks.
"""

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    fen: Optional[str] = Field(None, description="Chess position in FEN notation")
    puzzle_id: Optional[str] = Field(None, description="Associated puzzle ID")
    session_id: str = Field(..., description="User session identifier")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Conversation history")
    
    @validator("history")
    def validate_history(cls, v):
        """Validate conversation history format."""
        if not isinstance(v, list):
            return []
        
        for item in v:
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                raise ValueError("History items must have 'role' and 'content' fields")
            if item["role"] not in ["user", "assistant"]:
                raise ValueError("Role must be 'user' or 'assistant'")
        
        return v[-10:]  # Keep only last 10 messages
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What is the best move in this position?",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "session_id": "user_123",
                "history": [
                    {"role": "user", "content": "Can you analyze this position?"},
                    {"role": "assistant", "content": "This is the starting position..."}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    
    content: str = Field(..., description="Model response content")
    conversation_id: Optional[int] = Field(None, description="Stored conversation ID")
    model_version: str = Field(..., description="Model version used")
    response_time_ms: int = Field(..., description="Response generation time")
    chess_analysis: Optional[Dict[str, Any]] = Field(None, description="Chess-specific analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "This is the starting position in chess. Both sides have equal material...",
                "conversation_id": 123,
                "model_version": "v1.0",
                "response_time_ms": 1500,
                "chess_analysis": {
                    "position_type": "opening",
                    "material_balance": 0,
                    "suggested_moves": ["e4", "d4", "Nf3"]
                }
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    
    conversation_id: Optional[int] = Field(None, description="Conversation ID if available")
    session_id: str = Field(..., description="User session identifier")
    rating: int = Field(..., ge=-1, le=1, description="Rating: 1 (thumbs up), -1 (thumbs down)")
    comment: Optional[str] = Field(None, max_length=500, description="Optional feedback comment")
    message_content: Optional[str] = Field(None, description="Message content if no conversation_id")
    
    @validator("rating")
    def validate_rating(cls, v):
        """Validate rating values."""
        if v not in [-1, 1]:
            raise ValueError("Rating must be 1 (positive) or -1 (negative)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": 123,
                "session_id": "user_123",
                "rating": 1,
                "comment": "Great explanation of the position!"
            }
        }


class PuzzleResponse(BaseModel):
    """Response model for chess puzzles."""
    
    id: str = Field(..., description="Puzzle unique identifier")
    fen: str = Field(..., description="Position in FEN notation")
    unicode_position: Optional[str] = Field(None, description="Unicode representation")
    moves: str = Field(..., description="Solution moves")
    rating: Optional[int] = Field(None, description="Puzzle difficulty rating")
    themes: List[str] = Field(default=[], description="Puzzle themes/tags")
    popularity: Optional[int] = Field(None, description="Popularity score")
    game_url: Optional[str] = Field(None, description="Source game URL")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "puzzle_123",
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
                "moves": "Bxf7+ Ke7 Ng5",
                "rating": 1500,
                "themes": ["fork", "discovery", "short"],
                "popularity": 95
            }
        }


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str = Field(..., description="Overall health status")
    database: bool = Field(..., description="Database connection status")
    model: bool = Field(..., description="Model service status")
    chess_engine: bool = Field(..., description="Chess engine status")
    timestamp: float = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "database": True,
                "model": True,
                "chess_engine": True,
                "timestamp": 1640995200.0
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    max_length: int = Field(..., description="Maximum input length")
    ready: bool = Field(..., description="Model readiness status")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "chess-llm-v1",
                "version": "1.0.0",
                "max_length": 2048,
                "ready": True
            }
        }


class AnalysisRequest(BaseModel):
    """Request model for position analysis."""
    
    fen: str = Field(..., description="Position in FEN notation")
    depth: Optional[int] = Field(15, ge=1, le=30, description="Analysis depth")
    include_tactics: bool = Field(True, description="Include tactical analysis")
    include_evaluation: bool = Field(True, description="Include position evaluation")
    
    @validator("fen")
    def validate_fen(cls, v):
        """Validate FEN format."""
        try:
            import chess
            chess.Board(v)
        except Exception:
            raise ValueError("Invalid FEN notation")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "depth": 15,
                "include_tactics": True,
                "include_evaluation": True
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for position analysis."""
    
    fen: str = Field(..., description="Analyzed position")
    unicode_position: str = Field(..., description="Unicode representation")
    evaluation: Dict[str, Any] = Field(..., description="Position evaluation")
    legal_moves: List[str] = Field(..., description="Legal moves")
    best_moves: List[str] = Field(default=[], description="Engine recommended moves")
    tactics: Optional[Dict[str, Any]] = Field(None, description="Tactical patterns")
    analysis_time_ms: int = Field(..., description="Analysis duration")
    
    class Config:
        schema_extra = {
            "example": {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
                "evaluation": {"score": 0.0, "type": "centipawn"},
                "legal_moves": ["e4", "d4", "Nf3", "c4"],
                "best_moves": ["e4", "d4"],
                "tactics": {"checks": [], "captures": []},
                "analysis_time_ms": 250
            }
        }


class TrainingDataExport(BaseModel):
    """Training data export format."""
    
    input_text: str = Field(..., description="Training input")
    output_text: str = Field(..., description="Training target")
    quality_score: float = Field(..., description="Quality rating")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "input_text": "Analyze this chess position: ♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
                "output_text": "This is the starting position in chess with equal material for both sides...",
                "quality_score": 0.85,
                "metadata": {
                    "puzzle_id": "puzzle_123",
                    "user_rating": 1,
                    "themes": ["opening", "development"]
                }
            }
        }


class StatisticsResponse(BaseModel):
    """Application statistics response."""
    
    total_conversations: int = Field(..., description="Total conversations")
    positive_feedback: int = Field(..., description="Positive feedback count")
    negative_feedback: int = Field(..., description="Negative feedback count")
    unique_users: int = Field(..., description="Unique user sessions")
    puzzles_loaded: int = Field(..., description="Puzzles in database")
    training_examples: int = Field(..., description="Training examples generated")
    feedback_ratio: float = Field(..., description="Positive feedback percentage")
    uptime_seconds: float = Field(..., description="Application uptime")
    
    @validator("feedback_ratio", pre=True, always=True)
    def calculate_feedback_ratio(cls, v, values):
        """Calculate positive feedback ratio."""
        positive = values.get("positive_feedback", 0)
        negative = values.get("negative_feedback", 0)
        total = positive + negative
        return (positive / total * 100) if total > 0 else 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "total_conversations": 1250,
                "positive_feedback": 875,
                "negative_feedback": 125,
                "unique_users": 150,
                "puzzles_loaded": 50000,
                "training_examples": 800,
                "feedback_ratio": 87.5,
                "uptime_seconds": 86400
            }
        }


class QuestionRequest(BaseModel):
    """Request model for question generation."""
    
    level: int = Field(..., ge=1, le=5, description="Difficulty level (1-5)")
    fen: Optional[str] = Field(None, description="Specific chess position")
    puzzle_id: Optional[str] = Field(None, description="Use specific puzzle")
    question_type: Optional[str] = Field(None, description="Specific question type to generate")
    count: int = Field(1, ge=1, le=10, description="Number of questions to generate")
    
    @validator("fen")
    def validate_fen(cls, v):
        """Validate FEN format if provided."""
        if v is not None:
            try:
                import chess
                chess.Board(v)
            except Exception:
                raise ValueError("Invalid FEN notation")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "level": 2,
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "count": 3,
                "question_type": "piece_position"
            }
        }


class Question(BaseModel):
    """Individual generated question."""
    
    id: str = Field(..., description="Question unique identifier")
    level: int = Field(..., description="Difficulty level")
    question_type: str = Field(..., description="Type of question")
    question_text: str = Field(..., description="The question to ask")
    correct_answer: str = Field(..., description="Correct answer")
    alternative_answers: List[str] = Field(default=[], description="Alternative acceptable answers")
    fen: str = Field(..., description="Chess position for the question")
    unicode_position: str = Field(..., description="Unicode representation")
    explanation: Optional[str] = Field(None, description="Answer explanation")
    difficulty_rating: Optional[int] = Field(None, description="Specific difficulty rating")
    themes: List[str] = Field(default=[], description="Associated chess themes")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "q_level2_001",
                "level": 2,
                "question_type": "piece_position", 
                "question_text": "What piece is on the e1 square?",
                "correct_answer": "White King",
                "alternative_answers": ["King", "♔"],
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖",
                "explanation": "In chess starting position, the white king is placed on e1.",
                "themes": ["basics", "piece_identification"]
            }
        }


class QuestionResponse(BaseModel):
    """Response model for question generation."""
    
    questions: List[Question] = Field(..., description="Generated questions")
    generation_time_ms: int = Field(..., description="Time to generate questions")
    level: int = Field(..., description="Requested level")
    total_generated: int = Field(..., description="Number of questions generated")
    
    class Config:
        schema_extra = {
            "example": {
                "questions": [
                    {
                        "id": "q_level1_001",
                        "level": 1,
                        "question_type": "piece_count",
                        "question_text": "How many pawns does White have?",
                        "correct_answer": "8",
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖"
                    }
                ],
                "generation_time_ms": 150,
                "level": 1,
                "total_generated": 1
            }
        }


class QuestionValidationRequest(BaseModel):
    """Request model for question validation."""
    
    question: Question = Field(..., description="Question to validate")
    
    class Config:
        schema_extra = {
            "example": {
                "question": {
                    "id": "q_test_001",
                    "level": 1,
                    "question_type": "piece_count",
                    "question_text": "How many pawns does White have?",
                    "correct_answer": "8",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "unicode_position": "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖"
                }
            }
        }


class QuestionValidationResponse(BaseModel):
    """Response model for question validation."""
    
    is_valid: bool = Field(..., description="Whether question is valid")
    validation_score: float = Field(..., description="Validation score (0-1)")
    errors: List[str] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    chess_validity: bool = Field(..., description="Chess position validity")
    answer_correctness: bool = Field(..., description="Answer correctness")
    difficulty_match: bool = Field(..., description="Difficulty level match")
    
    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "validation_score": 0.95,
                "errors": [],
                "warnings": ["Answer could be more specific"],
                "chess_validity": True,
                "answer_correctness": True,
                "difficulty_match": True
            }
        }