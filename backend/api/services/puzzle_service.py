"""
Puzzle management service.

Handles retrieval and management of chess tactics puzzles from the database.
"""

import logging
import random
from typing import Dict, List, Optional

from sqlalchemy import and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models import TacticsPuzzle
from ..models import PuzzleResponse

logger = logging.getLogger(__name__)


class PuzzleService:
    """Service for managing chess tactics puzzles."""
    
    def __init__(self, database_manager):
        """Initialize puzzle service."""
        self.db = database_manager
    
    async def get_random_puzzle(
        self,
        difficulty: Optional[int] = None,
        themes: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        exclude_recent: bool = True
    ) -> Optional[PuzzleResponse]:
        """
        Get a random tactics puzzle based on criteria.
        
        Args:
            difficulty: Target difficulty rating
            themes: Preferred themes
            session_id: User session (to avoid recent puzzles)
            exclude_recent: Whether to exclude recently shown puzzles
            
        Returns:
            PuzzleResponse or None if no puzzles found
        """
        try:
            async with self.db.session_scope() as session:
                # Build query conditions
                conditions = []
                
                # Difficulty filtering
                if difficulty:
                    # Allow ±200 rating points around target
                    rating_range = 200
                    conditions.append(
                        and_(
                            TacticsPuzzle.rating >= difficulty - rating_range,
                            TacticsPuzzle.rating <= difficulty + rating_range
                        )
                    )
                
                # Theme filtering
                if themes:
                    # Match any of the requested themes
                    theme_conditions = []
                    for theme in themes:
                        theme_conditions.append(
                            TacticsPuzzle.themes.contains([theme])
                        )
                    if theme_conditions:
                        conditions.append(or_(*theme_conditions))
                
                # Exclude recent puzzles for this session
                if exclude_recent and session_id:
                    recent_puzzle_ids = await self._get_recent_puzzle_ids(session, session_id)
                    if recent_puzzle_ids:
                        conditions.append(
                            ~TacticsPuzzle.id.in_(recent_puzzle_ids)
                        )
                
                # Build query
                query = session.query(TacticsPuzzle)
                if conditions:
                    query = query.filter(and_(*conditions))
                
                # Order by popularity for better user experience
                query = query.order_by(TacticsPuzzle.popularity.desc())
                
                # Get total count
                total_count = await query.count()
                
                if total_count == 0:
                    logger.warning("No puzzles found matching criteria")
                    return None
                
                # Random selection from top results
                # Take top 20% by popularity, then random from those
                top_limit = max(10, total_count // 5)
                offset = random.randint(0, min(top_limit - 1, total_count - 1))
                
                puzzle = await query.offset(offset).first()
                
                if puzzle:
                    return self._puzzle_to_response(puzzle)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get random puzzle: {e}")
            return None
    
    async def get_puzzle_by_id(self, puzzle_id: str) -> Optional[PuzzleResponse]:
        """
        Get a specific puzzle by ID.
        
        Args:
            puzzle_id: Puzzle identifier
            
        Returns:
            PuzzleResponse or None if not found
        """
        try:
            async with self.db.session_scope() as session:
                puzzle = await session.query(TacticsPuzzle).filter(
                    TacticsPuzzle.id == puzzle_id
                ).first()
                
                if puzzle:
                    return self._puzzle_to_response(puzzle)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get puzzle {puzzle_id}: {e}")
            return None
    
    async def get_puzzles_by_theme(
        self,
        theme: str,
        difficulty_range: Optional[tuple] = None,
        limit: int = 10
    ) -> List[PuzzleResponse]:
        """
        Get puzzles by theme.
        
        Args:
            theme: Chess theme/tactic
            difficulty_range: (min_rating, max_rating) tuple
            limit: Maximum puzzles to return
            
        Returns:
            List of PuzzleResponse objects
        """
        try:
            async with self.db.session_scope() as session:
                query = session.query(TacticsPuzzle).filter(
                    TacticsPuzzle.themes.contains([theme])
                )
                
                # Apply difficulty range
                if difficulty_range:
                    min_rating, max_rating = difficulty_range
                    query = query.filter(
                        and_(
                            TacticsPuzzle.rating >= min_rating,
                            TacticsPuzzle.rating <= max_rating
                        )
                    )
                
                # Order by rating for progressive difficulty
                puzzles = await query.order_by(TacticsPuzzle.rating).limit(limit).all()
                
                return [self._puzzle_to_response(puzzle) for puzzle in puzzles]
                
        except Exception as e:
            logger.error(f"Failed to get puzzles by theme {theme}: {e}")
            return []
    
    async def get_puzzle_themes(self, limit: int = 50) -> List[Dict[str, any]]:
        """
        Get available puzzle themes with counts.
        
        Args:
            limit: Maximum themes to return
            
        Returns:
            List of theme dictionaries with counts
        """
        try:
            async with self.db.session_scope() as session:
                # This is a PostgreSQL-specific query for array aggregation
                result = await session.execute(
                    """
                    SELECT 
                        theme,
                        COUNT(*) as count,
                        AVG(rating) as avg_rating,
                        MIN(rating) as min_rating,
                        MAX(rating) as max_rating
                    FROM (
                        SELECT UNNEST(themes) as theme, rating
                        FROM tactics_puzzles
                        WHERE themes IS NOT NULL
                    ) AS theme_data
                    GROUP BY theme
                    ORDER BY count DESC
                    LIMIT :limit
                    """,
                    {"limit": limit}
                )
                
                themes = []
                for row in result.fetchall():
                    themes.append({
                        "theme": row.theme,
                        "count": row.count,
                        "avg_rating": round(row.avg_rating, 0) if row.avg_rating else 0,
                        "min_rating": row.min_rating,
                        "max_rating": row.max_rating
                    })
                
                return themes
                
        except Exception as e:
            logger.error(f"Failed to get puzzle themes: {e}")
            return []
    
    async def get_puzzle_statistics(self) -> Dict[str, any]:
        """
        Get overall puzzle statistics.
        
        Returns:
            Dictionary with puzzle statistics
        """
        try:
            async with self.db.session_scope() as session:
                # Total puzzles
                total_result = await session.execute(
                    "SELECT COUNT(*) as count FROM tactics_puzzles"
                )
                total_puzzles = total_result.scalar()
                
                # Rating distribution
                rating_result = await session.execute(
                    """
                    SELECT 
                        MIN(rating) as min_rating,
                        MAX(rating) as max_rating,
                        AVG(rating) as avg_rating,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rating) as median_rating
                    FROM tactics_puzzles
                    WHERE rating IS NOT NULL
                    """
                )
                rating_stats = rating_result.fetchone()
                
                # Difficulty level distribution
                difficulty_result = await session.execute(
                    """
                    SELECT 
                        difficulty_level,
                        COUNT(*) as count
                    FROM tactics_puzzles
                    GROUP BY difficulty_level
                    ORDER BY difficulty_level
                    """
                )
                difficulty_dist = {row.difficulty_level: row.count for row in difficulty_result.fetchall()}
                
                # Most popular themes
                popular_themes = await self.get_puzzle_themes(limit=10)
                
                return {
                    "total_puzzles": total_puzzles or 0,
                    "min_rating": rating_stats.min_rating if rating_stats else 0,
                    "max_rating": rating_stats.max_rating if rating_stats else 0,
                    "avg_rating": round(rating_stats.avg_rating, 0) if rating_stats and rating_stats.avg_rating else 0,
                    "median_rating": round(rating_stats.median_rating, 0) if rating_stats and rating_stats.median_rating else 0,
                    "difficulty_distribution": difficulty_dist,
                    "popular_themes": popular_themes[:5]
                }
                
        except Exception as e:
            logger.error(f"Failed to get puzzle statistics: {e}")
            return {}
    
    async def search_puzzles(
        self,
        query: str,
        difficulty_range: Optional[tuple] = None,
        themes: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[PuzzleResponse]:
        """
        Search puzzles by various criteria.
        
        Args:
            query: Search query (FEN, opening, etc.)
            difficulty_range: (min_rating, max_rating) tuple
            themes: List of themes to filter by
            limit: Maximum results to return
            
        Returns:
            List of matching puzzles
        """
        try:
            async with self.db.session_scope() as session:
                conditions = []
                
                # Text search in FEN or opening tags
                if query:
                    search_conditions = []
                    
                    # Search in FEN
                    search_conditions.append(TacticsPuzzle.fen.ilike(f"%{query}%"))
                    
                    # Search in opening tags
                    search_conditions.append(
                        func.array_to_string(TacticsPuzzle.opening_tags, ' ').ilike(f"%{query}%")
                    )
                    
                    conditions.append(or_(*search_conditions))
                
                # Difficulty range
                if difficulty_range:
                    min_rating, max_rating = difficulty_range
                    conditions.append(
                        and_(
                            TacticsPuzzle.rating >= min_rating,
                            TacticsPuzzle.rating <= max_rating
                        )
                    )
                
                # Theme filtering
                if themes:
                    theme_conditions = []
                    for theme in themes:
                        theme_conditions.append(TacticsPuzzle.themes.contains([theme]))
                    conditions.append(or_(*theme_conditions))
                
                # Build and execute query
                query_obj = session.query(TacticsPuzzle)
                if conditions:
                    query_obj = query_obj.filter(and_(*conditions))
                
                puzzles = await query_obj.order_by(
                    TacticsPuzzle.popularity.desc()
                ).limit(limit).all()
                
                return [self._puzzle_to_response(puzzle) for puzzle in puzzles]
                
        except Exception as e:
            logger.error(f"Failed to search puzzles: {e}")
            return []
    
    async def _get_recent_puzzle_ids(
        self,
        session: AsyncSession,
        session_id: str,
        hours_back: int = 24
    ) -> List[str]:
        """Get puzzle IDs shown to user recently."""
        try:
            result = await session.execute(
                """
                SELECT DISTINCT c.puzzle_id
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                WHERE u.session_id = :session_id
                AND c.puzzle_id IS NOT NULL
                AND c.created_at > NOW() - INTERVAL ':hours hours'
                """,
                {"session_id": session_id, "hours": hours_back}
            )
            
            return [row.puzzle_id for row in result.fetchall()]
            
        except Exception as e:
            logger.warning(f"Failed to get recent puzzles: {e}")
            return []
    
    def _puzzle_to_response(self, puzzle: TacticsPuzzle) -> PuzzleResponse:
        """Convert database puzzle to response model."""
        return PuzzleResponse(
            id=puzzle.id,
            fen=puzzle.fen,
            unicode_position=puzzle.unicode_position,
            moves=puzzle.moves,
            rating=puzzle.rating,
            themes=puzzle.themes or [],
            popularity=puzzle.popularity,
            game_url=puzzle.game_url
        )