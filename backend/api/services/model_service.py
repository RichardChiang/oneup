"""
Model serving service for chess language model.

Handles model loading, inference, and response generation with streaming support.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
from threading import Thread

from ...chess_utils import ChessConverter
from ..models import ChatResponse

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for serving chess language model.
    
    Provides both streaming and complete response generation
    with chess-specific context handling.
    """
    
    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize model service.
        
        Args:
            model_path: HuggingFace model path or local path
            max_length: Maximum sequence length
            device: Device to load model on (auto-detect if None)
            load_in_8bit: Enable 8-bit quantization for memory efficiency
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = device or self._get_device()
        self.load_in_8bit = load_in_8bit
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Service state
        self._ready = False
        self._model_version = "1.0.0"
        
        logger.info(f"ModelService initialized with path: {model_path}")
    
    def _get_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def initialize(self):
        """Initialize model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Add special tokens for chess
            special_tokens = {
                "additional_special_tokens": [
                    "<chess_position>", "</chess_position>",
                    "<analysis>", "</analysis>",
                    "<move>", "</move>",
                    "<evaluation>", "</evaluation>"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Resize token embeddings for new special tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)
            
            # Set up generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            self._ready = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model service is ready."""
        return self._ready and self.model is not None
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.model_path.split("/")[-1] if "/" in self.model_path else self.model_path
    
    @property
    def model_version(self) -> str:
        """Get model version."""
        return self._model_version
    
    def _format_chess_prompt(
        self,
        message: str,
        fen: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format input for chess-specific prompting.
        
        Args:
            message: User message
            fen: Chess position in FEN notation
            conversation_history: Previous conversation
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_parts.append(f"Human: {content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")
        
        # Add chess position if provided
        if fen:
            try:
                unicode_pos = ChessConverter.fen_to_unicode(fen)
                prompt_parts.append(f"<chess_position>{unicode_pos}</chess_position>")
            except Exception as e:
                logger.warning(f"Failed to convert FEN to unicode: {e}")
                prompt_parts.append(f"Position (FEN): {fen}")
        
        # Add current message
        prompt_parts.append(f"Human: {message}")
        prompt_parts.append("Assistant: <analysis>")
        
        return "\n\n".join(prompt_parts)
    
    async def generate_response_stream(
        self,
        message: str,
        fen: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from the model.
        
        Args:
            message: User message
            fen: Chess position in FEN notation
            conversation_history: Previous conversation
            session_id: User session identifier
            
        Yields:
            Generated text chunks
        """
        if not self.is_ready():
            raise RuntimeError("Model service not ready")
        
        try:
            # Format prompt
            prompt = self._format_chess_prompt(message, fen, conversation_history)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=30.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation config for streaming
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "streamer": streamer,
                "generation_config": self.generation_config,
            }
            
            # Start generation in separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream results
            generated_text = ""
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    yield new_text
                    
                    # Stop at end analysis tag
                    if "</analysis>" in generated_text:
                        break
            
            # Wait for thread to complete
            thread.join()
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: Failed to generate response - {str(e)}"
    
    async def generate_complete_response(
        self,
        message: str,
        fen: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Generate complete response from the model.
        
        Args:
            message: User message
            fen: Chess position in FEN notation
            conversation_history: Previous conversation
            session_id: User session identifier
            
        Returns:
            Complete chat response
        """
        if not self.is_ready():
            raise RuntimeError("Model service not ready")
        
        start_time = time.time()
        
        try:
            # Format prompt
            prompt = self._format_chess_prompt(message, fen, conversation_history)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=self.generation_config,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode response
            generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up response
            response_text = self._post_process_response(response_text)
            
            # Calculate timing
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Analyze chess content if position provided
            chess_analysis = None
            if fen:
                chess_analysis = self._analyze_chess_response(response_text, fen)
            
            return ChatResponse(
                content=response_text,
                model_version=self.model_version,
                response_time_ms=response_time_ms,
                chess_analysis=chess_analysis
            )
            
        except Exception as e:
            logger.error(f"Complete generation failed: {e}")
            # Return error response
            return ChatResponse(
                content=f"I apologize, but I encountered an error generating a response: {str(e)}",
                model_version=self.model_version,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _post_process_response(self, response: str) -> str:
        """
        Clean up and post-process model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned response text
        """
        # Remove analysis tags
        response = response.replace("<analysis>", "").replace("</analysis>", "")
        
        # Remove other special tokens
        for token in ["<chess_position>", "</chess_position>", "<move>", "</move>", 
                     "<evaluation>", "</evaluation>"]:
            response = response.replace(token, "")
        
        # Clean up whitespace
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split(". ")
        if len(sentences) > 1 and not sentences[-1].endswith("."):
            response = ". ".join(sentences[:-1]) + "."
        
        return response
    
    def _analyze_chess_response(self, response: str, fen: str) -> Dict[str, any]:
        """
        Analyze chess-specific aspects of the response.
        
        Args:
            response: Model response text
            fen: Chess position
            
        Returns:
            Chess analysis metadata
        """
        analysis = {
            "position_type": "unknown",
            "mentions_moves": False,
            "mentions_pieces": False,
            "mentions_tactics": False,
            "response_length": len(response),
            "chess_terms_count": 0
        }
        
        # Chess terminology detection
        chess_terms = [
            "pawn", "rook", "knight", "bishop", "queen", "king",
            "check", "checkmate", "castle", "en passant",
            "fork", "pin", "skewer", "discovery", "sacrifice",
            "attack", "defend", "capture", "promote",
            "opening", "middlegame", "endgame"
        ]
        
        response_lower = response.lower()
        for term in chess_terms:
            if term in response_lower:
                analysis["chess_terms_count"] += response_lower.count(term)
                
                if term in ["fork", "pin", "skewer", "discovery", "sacrifice"]:
                    analysis["mentions_tactics"] = True
                elif term in ["pawn", "rook", "knight", "bishop", "queen", "king"]:
                    analysis["mentions_pieces"] = True
        
        # Move notation detection
        import re
        move_patterns = [
            r"\b[a-h][1-8]\b",  # Square notation
            r"\b[NBRQK][a-h][1-8]\b",  # Piece moves
            r"\b[a-h]x[a-h][1-8]\b",  # Captures
            r"\bO-O(-O)?\b",  # Castling
        ]
        
        for pattern in move_patterns:
            if re.search(pattern, response):
                analysis["mentions_moves"] = True
                break
        
        return analysis
    
    async def cleanup(self):
        """Cleanup model resources."""
        try:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            
            self._ready = False
            logger.info("Model service cleanup completed")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")