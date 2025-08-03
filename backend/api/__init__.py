"""
FastAPI backend for chess RL training system.

Provides REST API for model serving, conversation management, and data collection.
"""

from .main import app

__all__ = ["app"]