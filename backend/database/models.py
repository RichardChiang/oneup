"""
SQLAlchemy models for chess RL training system.

Defines database schema for tactics puzzles, conversations, training data, and users.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class TacticsPuzzle(Base):
    """
    Lichess tactics puzzle data.
    
    Stores puzzle information including position, solution, and metadata.
    """
    __tablename__ = "tactics_puzzles"
    
    id = Column(String, primary_key=True)
    fen = Column(Text, nullable=False, index=True)
    moves = Column(Text, nullable=False)  # Solution moves
    rating = Column(Integer, index=True)
    rd = Column(Integer)  # Rating deviation
    popularity = Column(Integer, default=0)
    nb_plays = Column(Integer, default=0)
    themes = Column(ARRAY(String), index=True)
    game_url = Column(String)
    opening_tags = Column(ARRAY(String))
    
    # Computed fields
    unicode_position = Column(Text)  # Unicode representation
    difficulty_level = Column(Integer, index=True)  # 1-5 curriculum level
    move_count = Column(Integer)  # Number of moves in solution
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    conversations = relationship("Conversation", back_populates="puzzle")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_puzzles_rating_themes', 'rating', 'themes'),
        Index('idx_puzzles_difficulty_rating', 'difficulty_level', 'rating'),
    )
    
    def __repr__(self):
        return f"<TacticsPuzzle(id='{self.id}', rating={self.rating}, themes={self.themes})>"


class User(Base):
    """
    User session tracking for conversations.
    
    Anonymous users identified by session ID.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now())
    
    # User preferences
    preferred_difficulty = Column(Integer, default=1500)
    preferred_themes = Column(ARRAY(String))
    
    # Statistics
    total_conversations = Column(Integer, default=0)
    positive_ratings = Column(Integer, default=0)
    negative_ratings = Column(Integer, default=0)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    
    def __repr__(self):
        return f"<User(session_id='{self.session_id}', conversations={self.total_conversations})>"


class Conversation(Base):
    """
    Chat conversations between users and the chess model.
    
    Stores complete conversation context with feedback and ratings.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    puzzle_id = Column(String, ForeignKey("tactics_puzzles.id"), nullable=True, index=True)
    
    # Conversation content
    user_message = Column(Text, nullable=False)
    model_response = Column(Text, nullable=False)
    context_messages = Column(Text)  # JSON array of previous messages
    
    # Model metadata
    model_version = Column(String, default="base")
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    
    # User feedback
    feedback_rating = Column(Integer)  # 1 = thumbs up, -1 = thumbs down
    feedback_comment = Column(Text)
    feedback_timestamp = Column(DateTime)
    
    # Analysis results
    analysis_type = Column(String)  # "puzzle", "position", "general"
    chess_validity_score = Column(Float)  # Automated validity check
    
    # Quality metrics for training
    completeness_score = Column(Float)
    specificity_score = Column(Float)
    directional_accuracy = Column(Float)
    overall_quality = Column(Float)
    
    # Training data status
    exported_for_training = Column(Boolean, default=False)
    training_weight = Column(Float, default=1.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    puzzle = relationship("TacticsPuzzle", back_populates="conversations")
    training_data = relationship("TrainingData", back_populates="conversation")
    
    # Indexes for analytics queries
    __table_args__ = (
        Index('idx_conversations_feedback', 'feedback_rating', 'created_at'),
        Index('idx_conversations_quality', 'overall_quality', 'exported_for_training'),
        Index('idx_conversations_user_date', 'user_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, rating={self.feedback_rating}, quality={self.overall_quality})>"


class TrainingData(Base):
    """
    Processed training examples derived from conversations.
    
    Converts conversations to supervised fine-tuning format.
    """
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True)
    
    # Source conversation
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    
    # Training format
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    instruction = Column(Text)  # Optional instruction prefix
    
    # Quality and filtering
    quality_score = Column(Float, nullable=False, index=True)
    data_split = Column(String, default="train")  # train/val/test
    
    # Metadata
    format_version = Column(String, default="v1")
    preprocessing_applied = Column(ARRAY(String))  # List of preprocessing steps
    
    # Export tracking
    exported_to_hf = Column(Boolean, default=False)
    export_batch_id = Column(String)
    hf_dataset_id = Column(String)
    
    # Training results
    used_in_training = Column(Boolean, default=False)
    training_run_id = Column(String)
    loss_contribution = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    exported_at = Column(DateTime)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="training_data")
    
    # Indexes for training data queries
    __table_args__ = (
        Index('idx_training_quality_split', 'quality_score', 'data_split'),
        Index('idx_training_export_status', 'exported_to_hf', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TrainingData(id={self.id}, quality={self.quality_score}, split={self.data_split})>"


class ModelCheckpoint(Base):
    """
    Model checkpoint metadata for version tracking.
    """
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True)
    
    # Model information
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=False)
    
    # Training information
    training_run_id = Column(String)
    epoch = Column(Integer)
    step = Column(Integer)
    
    # Performance metrics
    validation_loss = Column(Float)
    training_loss = Column(Float)
    chess_accuracy = Column(Float)
    human_eval_score = Column(Float)
    
    # Model configuration
    base_model = Column(String)
    parameters_count = Column(Integer)
    config_json = Column(Text)  # JSON serialized config
    
    # Deployment status
    is_active = Column(Boolean, default=False)
    deployment_timestamp = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_version'),
        Index('idx_checkpoints_active', 'is_active', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ModelCheckpoint(name='{self.model_name}', version='{self.version}', active={self.is_active})>"


class ProgressiveLevel(Base):
    """
    Progressive training curriculum levels and advancement tracking.
    """
    __tablename__ = "progressive_levels"
    
    id = Column(Integer, primary_key=True)
    
    # Level definition
    level_number = Column(Integer, nullable=False, unique=True)
    level_name = Column(String, nullable=False)
    description = Column(Text)
    
    # Success criteria
    success_threshold = Column(Float, default=0.85)  # 85% success rate
    min_examples = Column(Integer, default=100)
    
    # Question generation parameters
    rating_min = Column(Integer)
    rating_max = Column(Integer)
    allowed_themes = Column(ARRAY(String))
    max_moves = Column(Integer)
    
    # Progress tracking
    current_model_score = Column(Float, default=0.0)
    examples_completed = Column(Integer, default=0)
    advancement_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ProgressiveLevel(level={self.level_number}, name='{self.level_name}', score={self.current_model_score})>"


# Database utility functions
def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Drop all tables in the database."""
    Base.metadata.drop_all(bind=engine)