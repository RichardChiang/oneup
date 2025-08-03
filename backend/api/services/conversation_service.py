"""
Conversation management service.

Handles storing, retrieving, and managing chess conversations and training data generation.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models import Conversation, User, TrainingData
from ..models import TrainingDataExport

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations and training data."""
    
    def __init__(self, database_manager):
        """Initialize conversation service."""
        self.db = database_manager
    
    async def store_conversation(
        self,
        session_id: str,
        user_message: str,
        model_response: str,
        puzzle_id: Optional[str] = None,
        context: Optional[List[Dict]] = None,
        model_version: str = "1.0.0"
    ) -> int:
        """
        Store a conversation in the database.
        
        Args:
            session_id: User session identifier
            user_message: User's message
            model_response: Model's response
            puzzle_id: Associated puzzle ID
            context: Conversation context
            model_version: Model version used
            
        Returns:
            Conversation ID
        """
        try:
            async with self.db.session_scope() as session:
                # Get or create user
                user = await self._get_or_create_user(session, session_id)
                
                # Create conversation
                conversation = Conversation(
                    user_id=user.id,
                    puzzle_id=puzzle_id,
                    user_message=user_message,
                    model_response=model_response,
                    context_messages=json.dumps(context) if context else None,
                    model_version=model_version
                )
                
                session.add(conversation)
                await session.flush()
                
                # Update user statistics
                user.total_conversations += 1
                user.last_active = datetime.utcnow()
                
                conversation_id = conversation.id
                
                logger.info(f"Stored conversation {conversation_id} for session {session_id}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            raise
    
    async def _get_or_create_user(self, session: AsyncSession, session_id: str) -> User:
        """Get existing user or create new one."""
        # Try to find existing user
        result = await session.execute(
            "SELECT * FROM users WHERE session_id = :session_id",
            {"session_id": session_id}
        )
        user = result.fetchone()
        
        if user:
            return User(**dict(user))
        
        # Create new user
        new_user = User(session_id=session_id)
        session.add(new_user)
        await session.flush()
        
        return new_user
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: User session identifier
            limit: Maximum conversations to return
            
        Returns:
            List of conversation dictionaries
        """
        try:
            async with self.db.session_scope() as session:
                query = """
                    SELECT c.*, u.session_id
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    WHERE u.session_id = :session_id
                    ORDER BY c.created_at DESC
                    LIMIT :limit
                """
                
                result = await session.execute(
                    query,
                    {"session_id": session_id, "limit": limit}
                )
                
                conversations = []
                for row in result.fetchall():
                    conv_dict = dict(row)
                    if conv_dict.get("context_messages"):
                        conv_dict["context_messages"] = json.loads(conv_dict["context_messages"])
                    conversations.append(conv_dict)
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def update_conversation_feedback(
        self,
        conversation_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> bool:
        """
        Update conversation with user feedback.
        
        Args:
            conversation_id: Conversation ID
            rating: Feedback rating (-1, 0, 1)
            comment: Optional feedback comment
            
        Returns:
            True if successful
        """
        try:
            async with self.db.session_scope() as session:
                # Update conversation
                await session.execute(
                    """
                    UPDATE conversations 
                    SET feedback_rating = :rating,
                        feedback_comment = :comment,
                        feedback_timestamp = :timestamp
                    WHERE id = :conversation_id
                    """,
                    {
                        "rating": rating,
                        "comment": comment,
                        "timestamp": datetime.utcnow(),
                        "conversation_id": conversation_id
                    }
                )
                
                # Update user statistics
                if rating > 0:
                    await session.execute(
                        """
                        UPDATE users 
                        SET positive_ratings = positive_ratings + 1
                        WHERE id = (
                            SELECT user_id FROM conversations WHERE id = :conversation_id
                        )
                        """,
                        {"conversation_id": conversation_id}
                    )
                elif rating < 0:
                    await session.execute(
                        """
                        UPDATE users 
                        SET negative_ratings = negative_ratings + 1
                        WHERE id = (
                            SELECT user_id FROM conversations WHERE id = :conversation_id
                        )
                        """,
                        {"conversation_id": conversation_id}
                    )
                
                logger.info(f"Updated feedback for conversation {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update feedback: {e}")
            return False
    
    async def generate_training_data(
        self,
        min_rating: int = 1,
        quality_threshold: float = 0.7,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Generate training data from high-quality conversations.
        
        Args:
            min_rating: Minimum feedback rating
            quality_threshold: Minimum quality score
            limit: Maximum examples to generate
            
        Returns:
            List of training examples
        """
        try:
            async with self.db.session_scope() as session:
                # Get high-quality conversations
                query = """
                    SELECT c.*, tp.unicode_position, tp.themes
                    FROM conversations c
                    LEFT JOIN tactics_puzzles tp ON c.puzzle_id = tp.id
                    WHERE c.feedback_rating >= :min_rating
                    AND (c.overall_quality IS NULL OR c.overall_quality >= :quality_threshold)
                    AND c.exported_for_training = FALSE
                    ORDER BY c.feedback_rating DESC, c.created_at DESC
                    LIMIT :limit
                """
                
                result = await session.execute(
                    query,
                    {
                        "min_rating": min_rating,
                        "quality_threshold": quality_threshold,
                        "limit": limit
                    }
                )
                
                training_examples = []
                for row in result.fetchall():
                    conv = dict(row)
                    
                    # Format training example
                    example = self._format_training_example(conv)
                    if example:
                        training_examples.append(example)
                        
                        # Mark as exported
                        await session.execute(
                            "UPDATE conversations SET exported_for_training = TRUE WHERE id = :id",
                            {"id": conv["id"]}
                        )
                
                logger.info(f"Generated {len(training_examples)} training examples")
                return training_examples
                
        except Exception as e:
            logger.error(f"Failed to generate training data: {e}")
            return []
    
    def _format_training_example(self, conversation: Dict) -> Optional[Dict]:
        """Format conversation as training example."""
        try:
            # Create input text
            input_parts = []
            
            # Add position if available
            if conversation.get("unicode_position"):
                input_parts.append(f"Chess position: {conversation['unicode_position']}")
            
            # Add user message
            input_parts.append(f"Question: {conversation['user_message']}")
            
            input_text = "\n".join(input_parts)
            
            # Output is the model response
            output_text = conversation["model_response"]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(conversation)
            
            # Metadata
            metadata = {
                "conversation_id": conversation["id"],
                "puzzle_id": conversation.get("puzzle_id"),
                "themes": conversation.get("themes", []),
                "feedback_rating": conversation.get("feedback_rating"),
                "model_version": conversation.get("model_version"),
                "created_at": conversation["created_at"].isoformat() if conversation.get("created_at") else None
            }
            
            return {
                "input_text": input_text,
                "output_text": output_text,
                "quality_score": quality_score,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to format training example: {e}")
            return None
    
    def _calculate_quality_score(self, conversation: Dict) -> float:
        """Calculate quality score for conversation."""
        score = 0.5  # Base score
        
        # Feedback rating contribution
        feedback_rating = conversation.get("feedback_rating", 0)
        if feedback_rating > 0:
            score += 0.3
        elif feedback_rating < 0:
            score -= 0.2
        
        # Response length (not too short, not too long)
        response_length = len(conversation.get("model_response", ""))
        if 100 <= response_length <= 1000:
            score += 0.1
        elif response_length < 50:
            score -= 0.2
        
        # Chess-specific content (basic heuristic)
        response = conversation.get("model_response", "").lower()
        chess_terms = ["move", "piece", "attack", "defend", "position", "tactic", "strategy"]
        chess_mentions = sum(1 for term in chess_terms if term in response)
        score += min(chess_mentions * 0.05, 0.2)
        
        return max(0.0, min(1.0, score))
    
    async def export_training_data(
        self,
        min_quality: float = 0.7,
        limit: int = 1000,
        format: str = "huggingface"
    ) -> List[TrainingDataExport]:
        """
        Export training data in specified format.
        
        Args:
            min_quality: Minimum quality threshold
            limit: Maximum examples to export
            format: Export format (huggingface, jsonl, etc.)
            
        Returns:
            List of training data exports
        """
        try:
            # Generate fresh training data
            examples = await self.generate_training_data(
                min_rating=1,
                quality_threshold=min_quality,
                limit=limit
            )
            
            # Convert to export format
            exports = []
            for example in examples:
                if format == "huggingface":
                    export = TrainingDataExport(
                        input_text=example["input_text"],
                        output_text=example["output_text"],
                        quality_score=example["quality_score"],
                        metadata=example["metadata"]
                    )
                    exports.append(export)
            
            logger.info(f"Exported {len(exports)} training examples in {format} format")
            return exports
            
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return []
    
    async def get_statistics(self, session: AsyncSession) -> Dict:
        """Get conversation statistics."""
        try:
            # Total conversations
            total_conv_result = await session.execute(
                "SELECT COUNT(*) as count FROM conversations"
            )
            total_conversations = total_conv_result.scalar()
            
            # Feedback statistics
            feedback_result = await session.execute(
                """
                SELECT 
                    SUM(CASE WHEN feedback_rating > 0 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN feedback_rating < 0 THEN 1 ELSE 0 END) as negative
                FROM conversations 
                WHERE feedback_rating IS NOT NULL
                """
            )
            feedback_row = feedback_result.fetchone()
            
            # Unique users
            users_result = await session.execute(
                "SELECT COUNT(DISTINCT session_id) as count FROM users"
            )
            unique_users = users_result.scalar()
            
            # Training examples
            training_result = await session.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE exported_for_training = TRUE"
            )
            training_examples = training_result.scalar()
            
            return {
                "total_conversations": total_conversations or 0,
                "positive_feedback": feedback_row.positive if feedback_row else 0,
                "negative_feedback": feedback_row.negative if feedback_row else 0,
                "unique_users": unique_users or 0,
                "training_examples": training_examples or 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}