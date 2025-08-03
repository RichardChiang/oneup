"""
Feedback management service.

Handles user feedback collection, processing, and quality assessment.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models import Conversation, User
from .conversation_service import ConversationService

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for managing user feedback and quality assessment."""
    
    def __init__(self, database_manager):
        """Initialize feedback service."""
        self.db = database_manager
        self.conversation_service = ConversationService(database_manager)
    
    async def submit_feedback(
        self,
        conversation_id: Optional[int] = None,
        session_id: Optional[str] = None,
        rating: int = 0,
        comment: str = "",
        message_content: Optional[str] = None
    ) -> bool:
        """
        Submit user feedback for a conversation.
        
        Args:
            conversation_id: Conversation ID (if available)
            session_id: User session ID
            rating: Feedback rating (-1, 0, 1)
            comment: Optional feedback comment
            message_content: Message content (fallback if no conversation_id)
            
        Returns:
            True if feedback submitted successfully
        """
        try:
            async with self.db.session_scope() as session:
                if conversation_id:
                    # Update existing conversation
                    success = await self.conversation_service.update_conversation_feedback(
                        conversation_id, rating, comment
                    )
                    
                    if success:
                        # Trigger quality assessment
                        await self._assess_conversation_quality(session, conversation_id)
                    
                    return success
                
                elif session_id and message_content:
                    # Find most recent conversation for this session with matching content
                    conversation = await self._find_conversation_by_content(
                        session, session_id, message_content
                    )
                    
                    if conversation:
                        return await self.submit_feedback(
                            conversation_id=conversation.id,
                            rating=rating,
                            comment=comment
                        )
                    else:
                        logger.warning(f"No conversation found for session {session_id}")
                        return False
                
                else:
                    logger.error("Insufficient information to submit feedback")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return False
    
    async def _find_conversation_by_content(
        self,
        session: AsyncSession,
        session_id: str,
        message_content: str
    ) -> Optional[Conversation]:
        """Find conversation by session and message content."""
        try:
            result = await session.execute(
                """
                SELECT c.* FROM conversations c
                JOIN users u ON c.user_id = u.id
                WHERE u.session_id = :session_id
                AND (c.user_message ILIKE :content OR c.model_response ILIKE :content)
                ORDER BY c.created_at DESC
                LIMIT 1
                """,
                {
                    "session_id": session_id,
                    "content": f"%{message_content[:100]}%"  # Match first 100 chars
                }
            )
            
            row = result.fetchone()
            if row:
                return Conversation(**dict(row))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find conversation: {e}")
            return None
    
    async def _assess_conversation_quality(
        self,
        session: AsyncSession,
        conversation_id: int
    ) -> Dict[str, float]:
        """
        Assess conversation quality using multiple criteria.
        
        Args:
            session: Database session
            conversation_id: Conversation to assess
            
        Returns:
            Dictionary with quality scores
        """
        try:
            # Get conversation details
            result = await session.execute(
                """
                SELECT c.*, tp.themes, tp.rating as puzzle_rating
                FROM conversations c
                LEFT JOIN tactics_puzzles tp ON c.puzzle_id = tp.id
                WHERE c.id = :conversation_id
                """,
                {"conversation_id": conversation_id}
            )
            
            conv_row = result.fetchone()
            if not conv_row:
                return {}
            
            conv = dict(conv_row)
            
            # Calculate quality scores
            quality_scores = {
                "completeness_score": self._assess_completeness(conv),
                "specificity_score": self._assess_specificity(conv),
                "directional_accuracy": self._assess_directional_accuracy(conv),
                "chess_validity": self._assess_chess_validity(conv)
            }
            
            # Calculate overall quality
            overall_quality = (
                0.3 * quality_scores["completeness_score"] +
                0.25 * quality_scores["specificity_score"] +
                0.25 * quality_scores["directional_accuracy"] +
                0.2 * quality_scores["chess_validity"]
            )
            
            quality_scores["overall_quality"] = overall_quality
            
            # Update conversation with quality scores
            await session.execute(
                """
                UPDATE conversations SET
                    completeness_score = :completeness,
                    specificity_score = :specificity,
                    directional_accuracy = :directional,
                    chess_validity_score = :chess_validity,
                    overall_quality = :overall
                WHERE id = :conversation_id
                """,
                {
                    "completeness": quality_scores["completeness_score"],
                    "specificity": quality_scores["specificity_score"],
                    "directional": quality_scores["directional_accuracy"],
                    "chess_validity": quality_scores["chess_validity"],
                    "overall": overall_quality,
                    "conversation_id": conversation_id
                }
            )
            
            logger.info(f"Quality assessment completed for conversation {conversation_id}")
            return quality_scores
            
        except Exception as e:
            logger.error(f"Failed to assess conversation quality: {e}")
            return {}
    
    def _assess_completeness(self, conversation: Dict) -> float:
        """
        Assess how completely the response addresses the question.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Completeness score (0.0-1.0)
        """
        user_message = conversation.get("user_message", "").lower()
        model_response = conversation.get("model_response", "").lower()
        
        score = 0.5  # Base score
        
        # Check if response length is appropriate
        response_length = len(model_response)
        if 100 <= response_length <= 1000:
            score += 0.2
        elif response_length < 50:
            score -= 0.3
        
        # Check for key question words and corresponding responses
        question_indicators = {
            "what": ["what", "this", "that", "answer", "explanation"],
            "why": ["because", "reason", "due to", "since"],
            "how": ["by", "through", "method", "way", "process"],
            "where": ["at", "on", "in", "position", "square"],
            "analyze": ["analysis", "evaluation", "assessment"],
            "best": ["best", "optimal", "strongest", "recommended"]
        }
        
        for question_word, response_words in question_indicators.items():
            if question_word in user_message:
                if any(word in model_response for word in response_words):
                    score += 0.1
                else:
                    score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_specificity(self, conversation: Dict) -> float:
        """
        Assess how specific and detailed the response is.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Specificity score (0.0-1.0)
        """
        model_response = conversation.get("model_response", "").lower()
        
        score = 0.3  # Base score
        
        # Check for specific chess terminology
        specific_terms = [
            # Pieces
            "pawn", "rook", "knight", "bishop", "queen", "king",
            # Squares
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
            "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
            "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
            "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
            "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8",
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
            "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
            "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8",
            # Tactics
            "fork", "pin", "skewer", "discovery", "deflection",
            "decoy", "clearance", "interference", "zugzwang",
            # Moves
            "castle", "en passant", "promote", "capture"
        ]
        
        specific_count = sum(1 for term in specific_terms if term in model_response)
        score += min(specific_count * 0.05, 0.4)
        
        # Penalize generic responses
        generic_phrases = [
            "good move", "bad move", "depends on", "it varies",
            "many possibilities", "hard to say", "not sure"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in model_response)
        score -= generic_count * 0.1
        
        # Check for concrete move suggestions
        import re
        move_pattern = r'\b[a-h][1-8]\b|\b[NBRQK][a-h][1-8]\b|\bO-O\b|\bO-O-O\b'
        moves_mentioned = len(re.findall(move_pattern, model_response))
        if moves_mentioned > 0:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_directional_accuracy(self, conversation: Dict) -> float:
        """
        Assess if the response is heading in the right direction.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Directional accuracy score (0.0-1.0)
        """
        score = 0.6  # Base score (neutral)
        
        # Use feedback rating as primary indicator
        feedback_rating = conversation.get("feedback_rating")
        if feedback_rating is not None:
            if feedback_rating > 0:
                score += 0.3
            elif feedback_rating < 0:
                score -= 0.4
        
        # Check for chess logic indicators
        model_response = conversation.get("model_response", "").lower()
        
        # Positive indicators
        positive_indicators = [
            "control", "development", "safety", "activity",
            "coordination", "tempo", "initiative", "advantage",
            "pressure", "weakness", "strength"
        ]
        
        for indicator in positive_indicators:
            if indicator in model_response:
                score += 0.05
        
        # Check for puzzle themes alignment
        puzzle_themes = conversation.get("themes", [])
        if puzzle_themes:
            theme_mentions = sum(1 for theme in puzzle_themes if theme.lower() in model_response)
            if theme_mentions > 0:
                score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _assess_chess_validity(self, conversation: Dict) -> float:
        """
        Assess if chess concepts are correctly applied.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Chess validity score (0.0-1.0)
        """
        model_response = conversation.get("model_response", "").lower()
        
        score = 0.7  # Base score (assume mostly valid)
        
        # Check for common chess misconceptions
        misconceptions = [
            "pawn moves backward",
            "king can move two squares",
            "castle through check",
            "castle when king moved",
            "en passant any time"
        ]
        
        for misconception in misconceptions:
            if misconception in model_response:
                score -= 0.2
        
        # Check for correct chess terminology usage
        correct_terms = [
            "check", "checkmate", "stalemate", "castle", "en passant",
            "promotion", "capture", "attack", "defend", "control"
        ]
        
        correct_usage = sum(1 for term in correct_terms if term in model_response)
        score += min(correct_usage * 0.03, 0.2)
        
        # Basic position validation if FEN is mentioned
        if "fen" in model_response:
            # Could add FEN validation here
            pass
        
        return max(0.0, min(1.0, score))
    
    async def get_feedback_statistics(self) -> Dict[str, any]:
        """Get feedback statistics and trends."""
        try:
            async with self.db.session_scope() as session:
                # Overall feedback stats
                result = await session.execute(
                    """
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN feedback_rating > 0 THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN feedback_rating < 0 THEN 1 ELSE 0 END) as negative,
                        AVG(CASE WHEN overall_quality IS NOT NULL THEN overall_quality END) as avg_quality
                    FROM conversations
                    WHERE feedback_rating IS NOT NULL
                    """
                )
                
                stats_row = result.fetchone()
                
                # Quality score distribution
                quality_result = await session.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN overall_quality >= 0.8 THEN 'excellent'
                            WHEN overall_quality >= 0.6 THEN 'good'
                            WHEN overall_quality >= 0.4 THEN 'fair'
                            ELSE 'poor'
                        END as quality_category,
                        COUNT(*) as count
                    FROM conversations
                    WHERE overall_quality IS NOT NULL
                    GROUP BY quality_category
                    """
                )
                
                quality_dist = {row.quality_category: row.count for row in quality_result.fetchall()}
                
                return {
                    "total_feedback": stats_row.total_feedback or 0,
                    "positive_feedback": stats_row.positive or 0,
                    "negative_feedback": stats_row.negative or 0,
                    "average_quality": round(stats_row.avg_quality, 3) if stats_row.avg_quality else 0,
                    "quality_distribution": quality_dist
                }
                
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {}