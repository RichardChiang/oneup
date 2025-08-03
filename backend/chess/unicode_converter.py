"""
Unicode chess representation converter.

Converts between FEN notation and unicode representation for optimal tokenization.
Each chess piece maps to a single unicode character, avoiding subword splitting.
"""

import re
from typing import Dict, List, Optional

import chess

# Unicode piece mappings
PIECE_TO_UNICODE: Dict[str, str] = {
    'K': '♔',  # White King
    'Q': '♕',  # White Queen
    'R': '♖',  # White Rook
    'B': '♗',  # White Bishop
    'N': '♘',  # White Knight
    'P': '♙',  # White Pawn
    'k': '♚',  # Black King
    'q': '♛',  # Black Queen
    'r': '♜',  # Black Rook
    'b': '♝',  # Black Bishop
    'n': '♞',  # Black Knight
    'p': '♟',  # Black Pawn
    '1': '□',  # Empty square
    '2': '□□',  # Two empty squares
    '3': '□□□',  # Three empty squares
    '4': '□□□□',  # Four empty squares
    '5': '□□□□□',  # Five empty squares
    '6': '□□□□□□',  # Six empty squares
    '7': '□□□□□□□',  # Seven empty squares
    '8': '□□□□□□□□',  # Eight empty squares
}

# Reverse mapping for conversion back to FEN
UNICODE_TO_PIECE: Dict[str, str] = {
    '♔': 'K', '♕': 'Q', '♖': 'R', '♗': 'B', '♘': 'N', '♙': 'P',
    '♚': 'k', '♛': 'q', '♜': 'r', '♝': 'b', '♞': 'n', '♟': 'p',
    '□': '1'
}


class ChessConversionError(Exception):
    """Raised when chess conversion fails."""
    pass


class ChessConverter:
    """
    Converts between FEN notation and unicode representation.
    
    This enables optimal tokenization where each piece = 1 token,
    improving model learning and pattern recognition.
    """
    
    @staticmethod
    def fen_to_unicode(fen: str) -> str:
        """
        Convert FEN position to unicode string representation.
        
        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            
        Returns:
            Unicode representation of the board position
            
        Raises:
            ChessConversionError: If FEN is invalid or conversion fails
        """
        try:
            # Validate FEN first
            board = chess.Board(fen)
            board_fen = board.board_fen()
            
            # Convert board position only (ignore game state)
            unicode_board = ""
            
            for char in board_fen:
                if char == '/':
                    continue  # Skip rank separators
                elif char in PIECE_TO_UNICODE:
                    unicode_board += PIECE_TO_UNICODE[char]
                else:
                    raise ChessConversionError(f"Unknown character in FEN: {char}")
            
            return unicode_board
            
        except chess.InvalidFenError as e:
            raise ChessConversionError(f"Invalid FEN: {e}")
        except Exception as e:
            raise ChessConversionError(f"Conversion failed: {e}")
    
    @staticmethod
    def unicode_to_fen(unicode_str: str) -> str:
        """
        Convert unicode string back to FEN notation.
        
        Args:
            unicode_str: Unicode representation of chess position
            
        Returns:
            FEN board position (board part only, no game state)
            
        Raises:
            ChessConversionError: If unicode string is invalid
        """
        try:
            if len(unicode_str) != 64:
                raise ChessConversionError(f"Unicode string must be 64 characters, got {len(unicode_str)}")
            
            # Convert unicode back to FEN
            fen_chars = []
            empty_count = 0
            
            for i, char in enumerate(unicode_str):
                # Add rank separator every 8 squares
                if i > 0 and i % 8 == 0:
                    if empty_count > 0:
                        fen_chars.append(str(empty_count))
                        empty_count = 0
                    fen_chars.append('/')
                
                if char == '□':
                    empty_count += 1
                elif char in UNICODE_TO_PIECE:
                    if empty_count > 0:
                        fen_chars.append(str(empty_count))
                        empty_count = 0
                    fen_chars.append(UNICODE_TO_PIECE[char])
                else:
                    raise ChessConversionError(f"Unknown unicode character: {char}")
            
            # Handle final empty squares
            if empty_count > 0:
                fen_chars.append(str(empty_count))
            
            board_fen = ''.join(fen_chars)
            
            # Validate by creating a board
            chess.Board(board_fen + " w - - 0 1")
            
            return board_fen
            
        except Exception as e:
            raise ChessConversionError(f"Unicode to FEN conversion failed: {e}")
    
    @staticmethod
    def get_board_unicode(board: chess.Board) -> str:
        """
        Get unicode representation directly from python-chess Board object.
        
        Args:
            board: python-chess Board object
            
        Returns:
            Unicode representation of the board
        """
        return ChessConverter.fen_to_unicode(board.fen())
    
    @staticmethod
    def validate_unicode(unicode_str: str) -> bool:
        """
        Validate that a unicode string represents a valid chess position.
        
        Args:
            unicode_str: Unicode representation to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            fen = ChessConverter.unicode_to_fen(unicode_str)
            chess.Board(fen + " w - - 0 1")
            return True
        except (ChessConversionError, chess.InvalidFenError):
            return False
    
    @staticmethod
    def count_pieces(unicode_str: str) -> Dict[str, int]:
        """
        Count pieces by type in unicode representation.
        
        Args:
            unicode_str: Unicode representation of chess position
            
        Returns:
            Dictionary mapping piece unicode to count
        """
        piece_counts = {}
        for char in unicode_str:
            if char != '□':  # Skip empty squares
                piece_counts[char] = piece_counts.get(char, 0) + 1
        return piece_counts
    
    @staticmethod
    def get_piece_at_square(unicode_str: str, square: str) -> Optional[str]:
        """
        Get the piece at a specific square.
        
        Args:
            unicode_str: Unicode representation of chess position
            square: Square in algebraic notation (e.g., "e4", "a1")
            
        Returns:
            Unicode character of piece at square, or None if empty
            
        Raises:
            ChessConversionError: If square notation is invalid
        """
        try:
            square_index = chess.parse_square(square)
            # Convert square index to unicode string index
            # chess.parse_square returns 0-63, we need to map this correctly
            rank = square_index // 8
            file = square_index % 8
            unicode_index = rank * 8 + file
            
            if 0 <= unicode_index < len(unicode_str):
                piece = unicode_str[unicode_index]
                return piece if piece != '□' else None
            else:
                raise ChessConversionError(f"Square index out of bounds: {square}")
                
        except ValueError as e:
            raise ChessConversionError(f"Invalid square notation: {square}")


def test_converter():
    """Test the chess converter with standard starting position."""
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    expected_unicode = "♜♞♝♛♚♝♞♜♟♟♟♟♟♟♟♟□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□♙♙♙♙♙♙♙♙♖♘♗♕♔♗♘♖"
    
    # Test FEN to Unicode
    unicode_result = ChessConverter.fen_to_unicode(starting_fen)
    assert unicode_result == expected_unicode, f"Expected {expected_unicode}, got {unicode_result}"
    
    # Test Unicode to FEN
    fen_result = ChessConverter.unicode_to_fen(unicode_result)
    expected_board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fen_result == expected_board_fen, f"Expected {expected_board_fen}, got {fen_result}"
    
    # Test validation
    assert ChessConverter.validate_unicode(unicode_result), "Unicode should be valid"
    
    # Test piece counting
    counts = ChessConverter.count_pieces(unicode_result)
    assert counts['♙'] == 8, "Should have 8 white pawns"
    assert counts['♟'] == 8, "Should have 8 black pawns"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_converter()