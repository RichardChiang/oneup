"""
FastAPI main application for chess RL training system.

Provides endpoints for model serving, conversation management, and puzzle retrieval.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ..database import initialize_database, get_database, get_session_scope
from ..chess import ChessConverter, ChessEngine
from .models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    PuzzleResponse,
    HealthResponse,
    ModelInfo,
)
from .services import (
    ModelService,
    ConversationService,
    PuzzleService,
    FeedbackService,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Database
    database_url: str = Field(default="postgresql://localhost/chess_rl")
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: List[str] = Field(default=["http://localhost:8501"])
    
    # Security
    secret_key: str = Field(default="dev-secret-key")
    
    # Model
    model_path: str = Field(default="microsoft/DialoGPT-medium")
    max_model_length: int = Field(default=2048)
    
    # Chess engine
    stockfish_path: Optional[str] = Field(default=None)
    stockfish_depth: int = Field(default=15)
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)
    
    # Development
    debug: bool = Field(default=False)
    reload: bool = Field(default=False)
    
    class Config:
        env_file = ".env"


# Global instances
settings = Settings()
model_service: Optional[ModelService] = None
conversation_service: Optional[ConversationService] = None
puzzle_service: Optional[PuzzleService] = None
feedback_service: Optional[FeedbackService] = None
chess_engine: Optional[ChessEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global model_service, conversation_service, puzzle_service, feedback_service, chess_engine
    
    logger.info("Starting chess RL training API...")
    
    try:
        # Initialize database
        db = initialize_database(settings.database_url, echo=settings.debug)
        await db.check_connection()
        logger.info("Database connection established")
        
        # Initialize chess engine
        chess_engine = ChessEngine(
            engine_path=settings.stockfish_path,
            depth=settings.stockfish_depth
        )
        logger.info("Chess engine initialized")
        
        # Initialize services
        model_service = ModelService(
            model_path=settings.model_path,
            max_length=settings.max_model_length
        )
        await model_service.initialize()
        
        conversation_service = ConversationService(db)
        puzzle_service = PuzzleService(db)
        feedback_service = FeedbackService(db)
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        
        if model_service:
            await model_service.cleanup()
        if chess_engine:
            chess_engine.close()
        
        db = get_database()
        await db.close()
        
        logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Chess RL Training API",
    description="Backend API for progressive chess understanding via self-evaluating language models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {duration:.3f}s with status {response.status_code}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db = get_database()
        db_healthy = await db.check_connection()
        
        # Check model service
        model_healthy = model_service is not None and model_service.is_ready()
        
        # Check chess engine
        engine_healthy = chess_engine is not None
        
        overall_healthy = db_healthy and model_healthy and engine_healthy
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            database=db_healthy,
            model=model_healthy,
            chess_engine=engine_healthy,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Model information endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    return ModelInfo(
        name=model_service.model_name,
        version=model_service.model_version,
        max_length=model_service.max_length,
        ready=model_service.is_ready()
    )


# Chat endpoint with streaming response
@app.post("/chat/response")
async def get_chess_response(request: ChatRequest) -> StreamingResponse:
    """
    Get chess model response for a given position and question.
    
    Supports streaming response for real-time user experience.
    """
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Validate chess position if provided
        if request.fen:
            if not ChessConverter.validate_unicode(ChessConverter.fen_to_unicode(request.fen)):
                raise HTTPException(status_code=400, detail="Invalid chess position")
        
        # Generate response stream
        response_stream = model_service.generate_response_stream(
            message=request.message,
            fen=request.fen,
            conversation_history=request.history,
            session_id=request.session_id
        )
        
        return StreamingResponse(
            response_stream,
            media_type="text/plain",
            headers={"X-Model-Version": model_service.model_version}
        )
        
    except Exception as e:
        logger.error(f"Chat response generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


# Non-streaming chat endpoint for simpler integration
@app.post("/chat/complete", response_model=ChatResponse)
async def get_complete_chess_response(request: ChatRequest):
    """Get complete chess model response (non-streaming)."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Generate complete response
        response = await model_service.generate_complete_response(
            message=request.message,
            fen=request.fen,
            conversation_history=request.history,
            session_id=request.session_id
        )
        
        # Store conversation
        if conversation_service:
            await conversation_service.store_conversation(
                session_id=request.session_id,
                user_message=request.message,
                model_response=response.content,
                puzzle_id=request.puzzle_id,
                context=request.history
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Complete chat response failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


# Feedback submission
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a conversation."""
    if not feedback_service:
        raise HTTPException(status_code=503, detail="Feedback service not available")
    
    try:
        await feedback_service.submit_feedback(
            conversation_id=feedback.conversation_id,
            rating=feedback.rating,
            comment=feedback.comment,
            session_id=feedback.session_id
        )
        
        return {"status": "success", "message": "Feedback submitted"}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


# Puzzle endpoints
@app.get("/puzzles/random", response_model=PuzzleResponse)
async def get_random_puzzle(
    difficulty: Optional[int] = None,
    themes: Optional[List[str]] = None,
    session_id: Optional[str] = None
):
    """Get a random tactics puzzle."""
    if not puzzle_service:
        raise HTTPException(status_code=503, detail="Puzzle service not available")
    
    try:
        puzzle = await puzzle_service.get_random_puzzle(
            difficulty=difficulty,
            themes=themes,
            session_id=session_id
        )
        
        if not puzzle:
            raise HTTPException(status_code=404, detail="No puzzles found matching criteria")
        
        return puzzle
        
    except Exception as e:
        logger.error(f"Random puzzle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get puzzle")


@app.get("/puzzles/{puzzle_id}", response_model=PuzzleResponse)
async def get_puzzle(puzzle_id: str):
    """Get a specific puzzle by ID."""
    if not puzzle_service:
        raise HTTPException(status_code=503, detail="Puzzle service not available")
    
    try:
        puzzle = await puzzle_service.get_puzzle_by_id(puzzle_id)
        
        if not puzzle:
            raise HTTPException(status_code=404, detail="Puzzle not found")
        
        return puzzle
        
    except Exception as e:
        logger.error(f"Puzzle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get puzzle")


# Chess position analysis
@app.post("/analysis/position")
async def analyze_position(fen: str):
    """Analyze a chess position using the engine."""
    if not chess_engine:
        raise HTTPException(status_code=503, detail="Chess engine not available")
    
    try:
        # Validate FEN
        if not ChessConverter.validate_unicode(ChessConverter.fen_to_unicode(fen)):
            raise HTTPException(status_code=400, detail="Invalid chess position")
        
        # Analyze position
        evaluation = chess_engine.evaluate_position(fen)
        legal_moves = chess_engine.get_legal_moves(fen)
        tactics = chess_engine.analyze_tactics(fen)
        
        return {
            "fen": fen,
            "unicode": ChessConverter.fen_to_unicode(fen),
            "evaluation": evaluation,
            "legal_moves": legal_moves,
            "tactics": tactics
        }
        
    except Exception as e:
        logger.error(f"Position analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze position")


# Training data export endpoint
@app.get("/training/export")
async def export_training_data(
    min_quality: float = 0.7,
    limit: int = 1000,
    format: str = "huggingface"
):
    """Export high-quality training data."""
    if not conversation_service:
        raise HTTPException(status_code=503, detail="Conversation service not available")
    
    try:
        training_data = await conversation_service.export_training_data(
            min_quality=min_quality,
            limit=limit,
            format=format
        )
        
        return {
            "data": training_data,
            "count": len(training_data),
            "format": format,
            "min_quality": min_quality
        }
        
    except Exception as e:
        logger.error(f"Training data export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to export training data")


# Statistics endpoint
@app.get("/stats")
async def get_statistics():
    """Get application statistics."""
    try:
        async with get_session_scope() as session:
            # Get basic statistics
            stats = await conversation_service.get_statistics(session)
            
            return {
                "total_conversations": stats.get("total_conversations", 0),
                "positive_feedback": stats.get("positive_feedback", 0),
                "negative_feedback": stats.get("negative_feedback", 0),
                "unique_users": stats.get("unique_users", 0),
                "puzzles_loaded": stats.get("puzzles_loaded", 0),
                "training_examples": stats.get("training_examples", 0),
                "uptime": time.time() - getattr(app.state, "start_time", time.time())
            }
            
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# Development endpoints (only in debug mode)
if settings.debug:
    @app.post("/dev/reset-db")
    async def reset_database():
        """Reset database (development only)."""
        try:
            db = get_database()
            db.drop_tables()
            db.create_tables()
            return {"status": "success", "message": "Database reset"}
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset database")


def run_server():
    """Run the FastAPI server."""
    app.state.start_time = time.time()
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level="info" if settings.debug else "warning",
    )


if __name__ == "__main__":
    run_server()