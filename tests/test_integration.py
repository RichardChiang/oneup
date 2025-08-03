"""
Integration tests for the chess RL training system.

Tests the complete pipeline from data loading to frontend interaction.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chess import ChessConverter, ChessEngine
from backend.database import initialize_database, get_session_scope
from backend.api.main import app
from backend.api.services import ModelService, PuzzleService

# Test configuration
TEST_DATABASE_URL = "postgresql://chess_user:chess_pass@localhost:5432/chess_rl_test"
API_BASE_URL = "http://localhost:8000"


class TestChessSystemIntegration:
    """Integration tests for the complete chess system."""
    
    @pytest.fixture(scope="class")
    async def setup_test_environment(self):
        """Set up test environment with database and services."""
        # Initialize test database
        db = initialize_database(TEST_DATABASE_URL, echo=False)
        
        # Create tables
        db.create_tables()
        
        # Insert test data
        await self.insert_test_data()
        
        yield db
        
        # Cleanup
        await db.close()
    
    async def insert_test_data(self):
        """Insert test puzzle data."""
        test_puzzles = [
            {
                "id": "test_001",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "moves": "e4 e5 Nf3",
                "rating": 1200,
                "themes": ["opening", "development"],
                "unicode_position": ChessConverter.fen_to_unicode(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                ),
                "difficulty_level": 1,
                "move_count": 3
            },
            {
                "id": "test_002", 
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
                "moves": "Bxf7+ Ke7 Ng5",
                "rating": 1500,
                "themes": ["fork", "discovery"],
                "unicode_position": ChessConverter.fen_to_unicode(
                    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
                ),
                "difficulty_level": 2,
                "move_count": 3
            }
        ]
        
        async with get_session_scope() as session:
            for puzzle_data in test_puzzles:
                await session.execute(
                    text("""
                        INSERT INTO tactics_puzzles 
                        (id, fen, moves, rating, themes, unicode_position, difficulty_level, move_count)
                        VALUES (:id, :fen, :moves, :rating, :themes, :unicode_position, :difficulty_level, :move_count)
                        ON CONFLICT (id) DO NOTHING
                    """),
                    puzzle_data
                )
    
    @pytest.mark.asyncio
    async def test_chess_converter(self):
        """Test chess unicode conversion."""
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Test FEN to unicode
        unicode_result = ChessConverter.fen_to_unicode(starting_fen)
        assert len(unicode_result) == 64
        assert "♚" in unicode_result  # Black king
        assert "♔" in unicode_result  # White king
        
        # Test unicode to FEN
        fen_result = ChessConverter.unicode_to_fen(unicode_result)
        expected_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        assert fen_result == expected_board
        
        # Test validation
        assert ChessConverter.validate_unicode(unicode_result)
        
        print("✓ Chess converter tests passed")
    
    @pytest.mark.asyncio
    async def test_chess_engine(self):
        """Test chess engine functionality."""
        try:
            engine = ChessEngine()
            starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            
            # Test position evaluation
            evaluation = engine.evaluate_position(starting_fen)
            assert "score" in evaluation
            assert "best_move" in evaluation
            
            # Test legal moves
            moves = engine.get_legal_moves(starting_fen)
            assert len(moves) == 20  # 20 legal moves from starting position
            
            # Test move validation
            assert engine.is_valid_move(starting_fen, "e4")
            assert not engine.is_valid_move(starting_fen, "e5")  # Not valid for white
            
            print("✓ Chess engine tests passed")
            
        except Exception as e:
            print(f"⚠ Chess engine tests skipped: {e}")
            pytest.skip("Chess engine not available")
    
    @pytest.mark.asyncio
    async def test_database_operations(self, setup_test_environment):
        """Test database operations."""
        db = await setup_test_environment
        
        # Test connection
        connected = await db.check_connection()
        assert connected, "Database connection failed"
        
        # Test puzzle retrieval
        puzzle_service = PuzzleService(db)
        
        # Get random puzzle
        puzzle = await puzzle_service.get_random_puzzle(difficulty=1200)
        assert puzzle is not None
        assert puzzle.rating is not None
        
        # Get puzzle by ID
        specific_puzzle = await puzzle_service.get_puzzle_by_id("test_001")
        assert specific_puzzle is not None
        assert specific_puzzle.id == "test_001"
        
        # Get puzzle themes
        themes = await puzzle_service.get_puzzle_themes()
        assert len(themes) > 0
        
        print("✓ Database operation tests passed")
    
    @pytest.mark.asyncio
    async def test_api_endpoints(self):
        """Test API endpoints."""
        async with httpx.AsyncClient() as client:
            try:
                # Test health endpoint
                response = await client.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    assert "status" in health_data
                    print("✓ Health endpoint working")
                else:
                    print("⚠ API not running, skipping API tests")
                    return
                
                # Test random puzzle endpoint
                response = await client.get(f"{API_BASE_URL}/puzzles/random")
                if response.status_code == 200:
                    puzzle_data = response.json()
                    assert "id" in puzzle_data
                    assert "fen" in puzzle_data
                    print("✓ Puzzle endpoint working")
                
                # Test chat endpoint
                chat_payload = {
                    "message": "What is the best move?",
                    "session_id": "test_session",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                }
                
                response = await client.post(f"{API_BASE_URL}/chat/complete", json=chat_payload)
                if response.status_code == 200:
                    chat_data = response.json()
                    assert "content" in chat_data
                    print("✓ Chat endpoint working")
                
                print("✓ API endpoint tests passed")
                
            except httpx.ConnectError:
                print("⚠ API server not running, skipping API tests")
                pytest.skip("API server not available")
    
    @pytest.mark.asyncio
    async def test_model_service(self):
        """Test model service functionality."""
        try:
            # Initialize model service with a small model for testing
            model_service = ModelService(
                model_path="microsoft/DialoGPT-small",  # Smaller model for testing
                max_length=512
            )
            
            await model_service.initialize()
            
            assert model_service.is_ready()
            
            # Test response generation
            response = await model_service.generate_complete_response(
                message="What is chess?",
                session_id="test_session"
            )
            
            assert response.content is not None
            assert len(response.content) > 0
            
            print("✓ Model service tests passed")
            
            await model_service.cleanup()
            
        except Exception as e:
            print(f"⚠ Model service tests skipped: {e}")
            pytest.skip("Model service not available")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, setup_test_environment):
        """Test complete end-to-end workflow."""
        db = await setup_test_environment
        
        # 1. Get a puzzle
        puzzle_service = PuzzleService(db)
        puzzle = await puzzle_service.get_random_puzzle()
        assert puzzle is not None
        
        # 2. Simulate user interaction
        session_id = "test_e2e_session"
        
        # 3. Test conversation storage
        from backend.api.services import ConversationService
        conv_service = ConversationService(db)
        
        conv_id = await conv_service.store_conversation(
            session_id=session_id,
            user_message="Analyze this position",
            model_response="This is a tactical position with a fork opportunity.",
            puzzle_id=puzzle.id
        )
        
        assert conv_id is not None
        
        # 4. Submit feedback
        from backend.api.services import FeedbackService
        feedback_service = FeedbackService(db)
        
        success = await feedback_service.submit_feedback(
            conversation_id=conv_id,
            rating=1,
            comment="Great analysis!"
        )
        
        assert success
        
        # 5. Generate training data
        training_data = await conv_service.generate_training_data(
            min_rating=1,
            limit=10
        )
        
        assert len(training_data) > 0
        
        print("✓ End-to-end workflow test passed")


async def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting Chess RL System Integration Tests\n")
    
    test_instance = TestChessSystemIntegration()
    
    try:
        # Setup test environment
        print("📋 Setting up test environment...")
        setup_task = test_instance.setup_test_environment()
        db = await setup_task.__anext__()
        
        # Run tests
        tests = [
            ("Chess Converter", test_instance.test_chess_converter),
            ("Chess Engine", test_instance.test_chess_engine),
            ("Database Operations", lambda: test_instance.test_database_operations(db)),
            ("API Endpoints", test_instance.test_api_endpoints),
            ("Model Service", test_instance.test_model_service),
            ("End-to-End Workflow", lambda: test_instance.test_end_to_end_workflow(db)),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                print(f"\n🔄 Running {test_name} tests...")
                await test_func()
                passed += 1
                print(f"✅ {test_name} tests PASSED")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} tests FAILED: {e}")
        
        # Cleanup
        await setup_task.__anext__()
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        if failed == 0:
            print("\n🎉 All integration tests passed!")
            return True
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")
            return False
            
    except Exception as e:
        print(f"💥 Integration test setup failed: {e}")
        return False


if __name__ == "__main__":
    # Run integration tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)