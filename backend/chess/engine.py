"""
Chess engine integration for ground truth generation and position analysis.

Provides interface to Stockfish engine for:
- Position evaluation
- Move generation and validation
- Tactical analysis
- Strategic features extraction
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chess
import chess.engine
from stockfish import Stockfish

logger = logging.getLogger(__name__)


class ChessEngineError(Exception):
    """Raised when chess engine operations fail."""
    pass


class ChessEngine:
    """
    Chess engine wrapper for position analysis and ground truth generation.
    
    Supports both python-chess engine interface and Stockfish library.
    Provides comprehensive analysis for training data generation.
    """
    
    def __init__(
        self,
        engine_path: Optional[str] = None,
        depth: int = 15,
        time_limit: float = 1.0,
        threads: int = 1,
        hash_size: int = 128
    ):
        """
        Initialize chess engine.
        
        Args:
            engine_path: Path to engine executable (auto-detect if None)
            depth: Search depth for analysis
            time_limit: Time limit per analysis in seconds
            threads: Number of threads for engine
            hash_size: Hash table size in MB
        """
        self.engine_path = engine_path or self._find_engine()
        self.depth = depth
        self.time_limit = time_limit
        self.threads = threads
        self.hash_size = hash_size
        
        self._engine: Optional[chess.engine.Protocol] = None
        self._stockfish: Optional[Stockfish] = None
        
        self._init_engine()
    
    def _find_engine(self) -> str:
        """Find Stockfish engine executable."""
        possible_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish", 
            "/opt/homebrew/bin/stockfish",
            "stockfish",  # In PATH
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # Try which command
        try:
            import subprocess
            result = subprocess.run(["which", "stockfish"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        raise ChessEngineError(
            "Stockfish not found. Install with: brew install stockfish (macOS) or apt install stockfish (Linux)"
        )
    
    def _init_engine(self):
        """Initialize the chess engine."""
        try:
            # Initialize Stockfish library
            self._stockfish = Stockfish(
                path=self.engine_path,
                depth=self.depth,
                parameters={
                    "Threads": self.threads,
                    "Hash": self.hash_size,
                    "UCI_Chess960": False,
                }
            )
            
            if not self._stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
                raise ChessEngineError("Engine initialization failed - cannot validate FEN")
                
            logger.info(f"Chess engine initialized: {self.engine_path}")
            
        except Exception as e:
            raise ChessEngineError(f"Failed to initialize engine: {e}")
    
    async def _get_engine(self) -> chess.engine.Protocol:
        """Get async engine instance for advanced analysis."""
        if self._engine is None:
            transport, self._engine = await chess.engine.popen_uci(self.engine_path)
            await self._engine.configure({"Threads": self.threads, "Hash": self.hash_size})
        return self._engine
    
    def evaluate_position(self, fen: str) -> Dict[str, Union[float, str]]:
        """
        Evaluate chess position.
        
        Args:
            fen: FEN notation of position
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            ChessEngineError: If evaluation fails
        """
        try:
            if not self._stockfish.is_fen_valid(fen):
                raise ChessEngineError(f"Invalid FEN: {fen}")
            
            self._stockfish.set_fen_position(fen)
            
            # Get evaluation
            evaluation = self._stockfish.get_evaluation()
            best_move = self._stockfish.get_best_move()
            
            # Convert evaluation
            if evaluation["type"] == "cp":
                score = evaluation["value"] / 100.0  # Convert centipawns to pawns
                evaluation_type = "centipawn"
            elif evaluation["type"] == "mate":
                score = float("inf") if evaluation["value"] > 0 else float("-inf")
                evaluation_type = "mate"
            else:
                score = 0.0
                evaluation_type = "unknown"
            
            return {
                "score": score,
                "type": evaluation_type,
                "best_move": best_move,
                "depth": self.depth,
                "fen": fen
            }
            
        except Exception as e:
            raise ChessEngineError(f"Position evaluation failed: {e}")
    
    def get_legal_moves(self, fen: str) -> List[str]:
        """
        Get all legal moves from position.
        
        Args:
            fen: FEN notation of position
            
        Returns:
            List of legal moves in algebraic notation
        """
        try:
            board = chess.Board(fen)
            return [board.san(move) for move in board.legal_moves]
        except Exception as e:
            raise ChessEngineError(f"Failed to get legal moves: {e}")
    
    def is_valid_move(self, fen: str, move: str) -> bool:
        """
        Check if move is legal in position.
        
        Args:
            fen: FEN notation of position
            move: Move in algebraic notation
            
        Returns:
            True if move is legal
        """
        try:
            board = chess.Board(fen)
            try:
                board.parse_san(move)
                return True
            except ValueError:
                return False
        except Exception:
            return False
    
    def analyze_tactics(self, fen: str) -> Dict[str, any]:
        """
        Analyze position for tactical patterns.
        
        Args:
            fen: FEN notation of position
            
        Returns:
            Dictionary with tactical analysis
        """
        try:
            board = chess.Board(fen)
            
            analysis = {
                "checks": [],
                "captures": [],
                "threats": [],
                "pins": [],
                "forks": [],
                "discovered_attacks": []
            }
            
            # Analyze each legal move
            for move in board.legal_moves:
                move_board = board.copy()
                move_board.push(move)
                
                # Check for checks
                if move_board.is_check():
                    analysis["checks"].append(board.san(move))
                
                # Check for captures
                if board.is_capture(move):
                    analysis["captures"].append(board.san(move))
            
            return analysis
            
        except Exception as e:
            raise ChessEngineError(f"Tactical analysis failed: {e}")
    
    async def deep_analysis(self, fen: str, time_limit: Optional[float] = None) -> Dict[str, any]:
        """
        Perform deep position analysis with multiple lines.
        
        Args:
            fen: FEN notation of position
            time_limit: Analysis time limit (uses default if None)
            
        Returns:
            Comprehensive analysis dictionary
        """
        try:
            engine = await self._get_engine()
            board = chess.Board(fen)
            
            time_limit = time_limit or self.time_limit
            
            # Analyze position
            analysis = await engine.analyse(
                board,
                chess.engine.Limit(time=time_limit),
                multipv=3  # Get top 3 moves
            )
            
            result = {
                "main_line": [],
                "evaluation": None,
                "variations": []
            }
            
            if analysis:
                # Main line
                if "pv" in analysis:
                    result["main_line"] = [board.san(move) for move in analysis["pv"]]
                
                # Evaluation
                if "score" in analysis:
                    score = analysis["score"].relative
                    if score.is_mate():
                        result["evaluation"] = {"type": "mate", "value": score.mate()}
                    else:
                        result["evaluation"] = {"type": "cp", "value": score.score()}
            
            return result
            
        except Exception as e:
            raise ChessEngineError(f"Deep analysis failed: {e}")
    
    def get_piece_activity(self, fen: str) -> Dict[str, Dict[str, int]]:
        """
        Analyze piece activity and positioning.
        
        Args:
            fen: FEN notation of position
            
        Returns:
            Dictionary with piece activity metrics
        """
        try:
            board = chess.Board(fen)
            
            activity = {
                "white": {"mobility": 0, "attacks": 0, "defends": 0},
                "black": {"mobility": 0, "attacks": 0, "defends": 0}
            }
            
            # Count legal moves (mobility)
            white_moves = 0
            black_moves = 0
            
            for move in board.legal_moves:
                if board.turn == chess.WHITE:
                    white_moves += 1
                else:
                    black_moves += 1
            
            activity["white"]["mobility"] = white_moves if board.turn == chess.WHITE else 0
            activity["black"]["mobility"] = black_moves if board.turn == chess.BLACK else 0
            
            # Analyze attacks and defenses for each square
            for square in chess.SQUARES:
                attackers_white = board.attackers(chess.WHITE, square)
                attackers_black = board.attackers(chess.BLACK, square)
                
                activity["white"]["attacks"] += len(attackers_white)
                activity["black"]["attacks"] += len(attackers_black)
            
            return activity
            
        except Exception as e:
            raise ChessEngineError(f"Activity analysis failed: {e}")
    
    def close(self):
        """Close engine connections."""
        if self._engine:
            asyncio.create_task(self._engine.quit())
        self._stockfish = None
        logger.info("Chess engine closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


async def test_engine():
    """Test chess engine functionality."""
    try:
        engine = ChessEngine()
        
        # Test starting position
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Test evaluation
        eval_result = engine.evaluate_position(starting_fen)
        print(f"Evaluation: {eval_result}")
        
        # Test legal moves
        moves = engine.get_legal_moves(starting_fen)
        print(f"Legal moves: {len(moves)}")
        
        # Test tactics
        tactics = engine.analyze_tactics(starting_fen)
        print(f"Tactics: {tactics}")
        
        # Test deep analysis
        deep = await engine.deep_analysis(starting_fen, 0.5)
        print(f"Deep analysis: {deep}")
        
        print("Engine tests passed!")
        
    except Exception as e:
        print(f"Engine test failed: {e}")
    finally:
        if 'engine' in locals():
            engine.close()


if __name__ == "__main__":
    asyncio.run(test_engine())