"""
Streamlit chess chat interface for human-in-the-loop training.

Provides an interactive chat experience with chess board visualization,
user feedback collection, and real-time model interaction.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import chess
import chess.svg
import httpx
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0

# Page config
st.set_page_config(
    page_title="Chess AI Trainer",
    page_icon="♔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chess-board {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    .feedback-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 10px 0;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    
    .chat-message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
    }
    
    .error-message {
        background-color: #ffebee;
        color: #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


class ChessInterface:
    """Main chess chat interface class."""
    
    def __init__(self):
        """Initialize the chess interface."""
        self.api_client = httpx.AsyncClient(base_url=API_BASE_URL, timeout=TIMEOUT)
        self.session_id = self._get_session_id()
        
        # Initialize session state
        self._init_session_state()
    
    def _get_session_id(self) -> str:
        """Get or create session ID."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_puzzle" not in st.session_state:
            st.session_state.current_puzzle = None
        
        if "feedback_pending" not in st.session_state:
            st.session_state.feedback_pending = {}
        
        if "statistics" not in st.session_state:
            st.session_state.statistics = {
                "total_conversations": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            }
        
        if "board_position" not in st.session_state:
            st.session_state.board_position = None
    
    async def get_random_puzzle(self, difficulty: Optional[int] = None) -> Optional[Dict]:
        """Fetch a random puzzle from the API."""
        try:
            params = {}
            if difficulty:
                params["difficulty"] = difficulty
            
            response = await self.api_client.get("/puzzles/random", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch puzzle: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching puzzle: {e}")
            st.error(f"Error fetching puzzle: {e}")
            return None
    
    async def send_message(self, message: str, fen: Optional[str] = None) -> Optional[str]:
        """Send message to the chess AI and get response."""
        try:
            payload = {
                "message": message,
                "session_id": self.session_id,
                "history": st.session_state.messages[-10:],  # Last 10 messages
            }
            
            if fen:
                payload["fen"] = fen
                
            if st.session_state.current_puzzle:
                payload["puzzle_id"] = st.session_state.current_puzzle["id"]
            
            response = await self.api_client.post("/chat/complete", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["content"], result.get("conversation_id")
            else:
                st.error(f"API Error: {response.status_code}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            st.error(f"Error sending message: {e}")
            return None, None
    
    async def submit_feedback(self, conversation_id: Optional[int], rating: int, comment: str = ""):
        """Submit user feedback."""
        try:
            payload = {
                "session_id": self.session_id,
                "rating": rating,
                "comment": comment
            }
            
            if conversation_id:
                payload["conversation_id"] = conversation_id
            else:
                # Fallback: use last message content
                if st.session_state.messages:
                    payload["message_content"] = st.session_state.messages[-1]["content"]
            
            response = await self.api_client.post("/feedback", json=payload)
            
            if response.status_code == 200:
                # Update statistics
                if rating > 0:
                    st.session_state.statistics["positive_feedback"] += 1
                else:
                    st.session_state.statistics["negative_feedback"] += 1
                
                st.success("Feedback submitted successfully!")
            else:
                st.error(f"Failed to submit feedback: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            st.error(f"Error submitting feedback: {e}")
    
    async def get_statistics(self) -> Dict:
        """Get application statistics."""
        try:
            response = await self.api_client.get("/stats")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}
    
    def render_chess_board(self, fen: str, size: int = 400) -> str:
        """Render chess board from FEN position."""
        try:
            board = chess.Board(fen)
            svg = chess.svg.board(
                board=board,
                size=size,
                style=""
            )
            return svg
        except Exception as e:
            logger.error(f"Error rendering board: {e}")
            return f"<p>Error rendering board: {e}</p>"
    
    def display_puzzle_info(self, puzzle: Dict):
        """Display puzzle information."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rating", puzzle.get("rating", "Unknown"))
        
        with col2:
            st.metric("Popularity", f"{puzzle.get('popularity', 0)}%")
        
        with col3:
            themes = puzzle.get("themes", [])
            st.metric("Themes", len(themes))
        
        if themes:
            st.write("**Themes:** " + ", ".join(themes[:5]))
    
    def render_feedback_buttons(self, conversation_id: Optional[int] = None):
        """Render thumbs up/down feedback buttons."""
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("👍", key=f"up_{conversation_id}_{time.time()}", 
                        help="Good response"):
                asyncio.run(self.submit_feedback(conversation_id, 1, "Helpful response"))
        
        with col2:
            if st.button("👎", key=f"down_{conversation_id}_{time.time()}", 
                        help="Poor response"):
                asyncio.run(self.submit_feedback(conversation_id, -1, "Could be better"))
        
        with col3:
            # Optional comment
            comment = st.text_input("Optional feedback comment", 
                                  key=f"comment_{conversation_id}_{time.time()}")
            if comment and st.button("Submit Comment", 
                                    key=f"submit_{conversation_id}_{time.time()}"):
                asyncio.run(self.submit_feedback(conversation_id, 0, comment))
    
    def run(self):
        """Main application interface."""
        # Header
        st.title("♔ Chess AI Trainer")
        st.markdown("Interactive chess training with AI feedback and human-in-the-loop learning")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            
            # Puzzle selection
            st.subheader("🎯 Puzzle Settings")
            
            difficulty_range = st.select_slider(
                "Difficulty Range",
                options=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
                value=(1200, 1800),
                help="Rating range for puzzles"
            )
            
            if st.button("🎲 Get New Puzzle", use_container_width=True):
                with st.spinner("Fetching puzzle..."):
                    # Use average of range as difficulty
                    avg_difficulty = sum(difficulty_range) // 2
                    puzzle = asyncio.run(self.get_random_puzzle(avg_difficulty))
                    
                    if puzzle:
                        st.session_state.current_puzzle = puzzle
                        st.session_state.board_position = puzzle["fen"]
                        st.success("New puzzle loaded!")
                        st.rerun()
            
            # Statistics
            st.subheader("📊 Statistics")
            stats = asyncio.run(self.get_statistics())
            
            if stats:
                st.metric("Total Conversations", stats.get("total_conversations", 0))
                st.metric("Positive Feedback", stats.get("positive_feedback", 0))
                st.metric("Success Rate", f"{stats.get('feedback_ratio', 0):.1f}%")
            
            # Session info
            st.subheader("🔧 Session Info")
            st.text(f"Session: {self.session_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_puzzle = None
                st.session_state.board_position = None
                st.rerun()
        
        # Main content area
        main_col1, main_col2 = st.columns([1, 1])
        
        with main_col1:
            st.subheader("♟️ Chess Position")
            
            # Display current puzzle/position
            if st.session_state.current_puzzle:
                puzzle = st.session_state.current_puzzle
                
                # Puzzle info
                self.display_puzzle_info(puzzle)
                
                # Chess board
                board_svg = self.render_chess_board(puzzle["fen"])
                st.markdown(
                    f'<div class="chess-board">{board_svg}</div>', 
                    unsafe_allow_html=True
                )
                
                # Solution (collapsible)
                with st.expander("💡 Show Solution", expanded=False):
                    st.write(f"**Moves:** {puzzle['moves']}")
                    if puzzle.get("game_url"):
                        st.markdown(f"[View Game]({puzzle['game_url']})")
            
            elif st.session_state.board_position:
                # Custom position
                board_svg = self.render_chess_board(st.session_state.board_position)
                st.markdown(
                    f'<div class="chess-board">{board_svg}</div>', 
                    unsafe_allow_html=True
                )
            
            else:
                # Default starting position
                starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                board_svg = self.render_chess_board(starting_fen)
                st.markdown(
                    f'<div class="chess-board">{board_svg}</div>', 
                    unsafe_allow_html=True
                )
                st.info("Load a puzzle or enter a FEN position to start analyzing!")
            
            # Manual FEN input
            st.subheader("🎯 Custom Position")
            fen_input = st.text_input(
                "Enter FEN notation:",
                placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                help="Enter a chess position in FEN notation"
            )
            
            if fen_input and st.button("Load Position"):
                try:
                    # Validate FEN
                    chess.Board(fen_input)
                    st.session_state.board_position = fen_input
                    st.session_state.current_puzzle = None  # Clear puzzle
                    st.success("Position loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid FEN: {e}")
        
        with main_col2:
            st.subheader("💬 Chat with Chess AI")
            
            # Display conversation history
            chat_container = st.container()
            
            with chat_container:
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        st.markdown(
                            f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-message assistant-message">🤖 **AI:** {message["content"]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Add feedback buttons for AI responses
                        conversation_id = message.get("conversation_id")
                        self.render_feedback_buttons(conversation_id)
                        st.divider()
            
            # Chat input
            st.subheader("💭 Ask about the position")
            
            # Suggested prompts
            if st.session_state.current_puzzle:
                st.write("**Quick prompts:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Analyze position", use_container_width=True):
                        st.session_state.temp_message = "Please analyze this chess position in detail."
                    
                    if st.button("🎯 Find the tactic", use_container_width=True):
                        st.session_state.temp_message = "What tactical opportunity exists in this position?"
                
                with col2:
                    if st.button("📋 Explain themes", use_container_width=True):
                        themes = st.session_state.current_puzzle.get("themes", [])
                        st.session_state.temp_message = f"Explain these chess themes: {', '.join(themes[:3])}"
                    
                    if st.button("🎓 Best move?", use_container_width=True):
                        st.session_state.temp_message = "What is the best move in this position and why?"
            
            # Main input
            user_input = st.text_area(
                "Your message:",
                value=getattr(st.session_state, 'temp_message', ''),
                placeholder="Ask about the position, request analysis, or discuss chess concepts...",
                height=100,
                key="user_message_input"
            )
            
            # Clear temp message after using it
            if hasattr(st.session_state, 'temp_message'):
                del st.session_state.temp_message
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                send_button = st.button("🚀 Send Message", use_container_width=True, type="primary")
            
            with col2:
                include_position = st.checkbox("Include position", value=True, 
                                             disabled=not st.session_state.current_puzzle and not st.session_state.board_position)
            
            # Handle message sending
            if send_button and user_input.strip():
                with st.spinner("🤔 AI is thinking..."):
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get current position if requested
                    current_fen = None
                    if include_position:
                        if st.session_state.current_puzzle:
                            current_fen = st.session_state.current_puzzle["fen"]
                        elif st.session_state.board_position:
                            current_fen = st.session_state.board_position
                    
                    # Send to AI
                    response, conversation_id = asyncio.run(
                        self.send_message(user_input, current_fen)
                    )
                    
                    if response:
                        # Add AI response
                        ai_message = {
                            "role": "assistant",
                            "content": response
                        }
                        if conversation_id:
                            ai_message["conversation_id"] = conversation_id
                        
                        st.session_state.messages.append(ai_message)
                        
                        # Update statistics
                        st.session_state.statistics["total_conversations"] += 1
                        
                        # Clear input and rerun
                        st.rerun()
                    else:
                        st.error("Failed to get response from AI")
            
            # Usage tips
            with st.expander("💡 Usage Tips", expanded=False):
                st.markdown("""
                **Great questions to ask:**
                - "What is the best move here and why?"
                - "Explain the tactical pattern in this position"
                - "What are the key strategic elements?"
                - "How should I approach this endgame?"
                - "What mistakes should I avoid?"
                
                **Features:**
                - 👍👎 Rate AI responses to improve the model
                - 🎲 Load random puzzles by difficulty
                - 🎯 Input custom positions via FEN
                - 💬 Full conversation history maintained
                """)


def main():
    """Main application entry point."""
    try:
        interface = ChessInterface()
        interface.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()