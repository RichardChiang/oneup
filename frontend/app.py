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
        
        # Practice mode state
        if "practice_mode" not in st.session_state:
            st.session_state.practice_mode = False
        
        if "current_questions" not in st.session_state:
            st.session_state.current_questions = []
        
        if "current_question_index" not in st.session_state:
            st.session_state.current_question_index = 0
        
        if "practice_level" not in st.session_state:
            st.session_state.practice_level = 1
        
        if "practice_score" not in st.session_state:
            st.session_state.practice_score = {"correct": 0, "total": 0}
        
        if "answer_submitted" not in st.session_state:
            st.session_state.answer_submitted = False
    
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
    
    async def generate_questions(self, level: int, count: int = 3, fen: Optional[str] = None) -> Optional[List[Dict]]:
        """Generate practice questions for the given level."""
        try:
            payload = {
                "level": level,
                "count": count
            }
            
            if fen:
                payload["fen"] = fen
            
            response = await self.api_client.post("/questions/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["questions"]
            else:
                st.error(f"Failed to generate questions: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            st.error(f"Error generating questions: {e}")
            return None
    
    async def get_question_levels(self) -> Optional[Dict]:
        """Get available question levels and types."""
        try:
            response = await self.api_client.get("/questions/levels")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error fetching question levels: {e}")
            return None
    
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
        
        # Main tabs
        tab1, tab2 = st.tabs(["💬 Chat Mode", "🎯 Practice Mode"])
        
        with tab1:
            self.render_chat_mode()
        
        with tab2:
            self.render_practice_mode()
    
    def render_chat_mode(self):
        """Render the chat interface mode."""
        
        # Sidebar
        with st.sidebar:
            st.header("Chat Controls")
            
            # Puzzle selection
            st.subheader("🎯 Puzzle Settings")
            
            difficulty_range = st.select_slider(
                "Difficulty Range",
                options=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
                value=(1200, 1800),
                help="Rating range for puzzles"
            )
            
            if st.button("🎲 Get New Puzzle", use_container_width=True, key="chat_new_puzzle"):
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
            
            if st.button("🗑️ Clear History", use_container_width=True, key="chat_clear_history"):
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
                help="Enter a chess position in FEN notation",
                key="chat_fen_input"
            )
            
            if fen_input and st.button("Load Position", key="chat_load_position"):
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
                    if st.button("🔍 Analyze position", use_container_width=True, key="chat_analyze"):
                        st.session_state.temp_message = "Please analyze this chess position in detail."
                    
                    if st.button("🎯 Find the tactic", use_container_width=True, key="chat_tactic"):
                        st.session_state.temp_message = "What tactical opportunity exists in this position?"
                
                with col2:
                    if st.button("📋 Explain themes", use_container_width=True, key="chat_themes"):
                        themes = st.session_state.current_puzzle.get("themes", [])
                        st.session_state.temp_message = f"Explain these chess themes: {', '.join(themes[:3])}"
                    
                    if st.button("🎓 Best move?", use_container_width=True, key="chat_best_move"):
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
                send_button = st.button("🚀 Send Message", use_container_width=True, type="primary", key="chat_send")
            
            with col2:
                include_position = st.checkbox("Include position", value=True, 
                                             disabled=not st.session_state.current_puzzle and not st.session_state.board_position,
                                             key="chat_include_position")
            
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
    
    def render_practice_mode(self):
        """Render the practice mode interface with generated questions."""
        
        # Sidebar for practice mode
        with st.sidebar:
            st.header("Practice Controls")
            
            # Level selection
            st.subheader("🎯 Difficulty Level")
            
            level_descriptions = {
                1: "Level 1: Piece Counting",
                2: "Level 2: Position Identification", 
                3: "Level 3: Basic Tactics",
                4: "Level 4: Strategic Analysis",
                5: "Level 5: Complex Reasoning"
            }
            
            selected_level = st.selectbox(
                "Choose difficulty:",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: level_descriptions[x],
                index=st.session_state.practice_level - 1,
                key="practice_level_select"
            )
            
            if selected_level != st.session_state.practice_level:
                st.session_state.practice_level = selected_level
                st.session_state.current_questions = []
                st.session_state.current_question_index = 0
                st.session_state.answer_submitted = False
            
            # Questions per session
            questions_count = st.slider("Questions per session:", 1, 10, 5, key="practice_questions_count")
            
            # Generate questions button
            if st.button("🚀 Start Practice Session", use_container_width=True, key="practice_start"):
                with st.spinner("Generating questions..."):
                    questions = asyncio.run(self.generate_questions(selected_level, questions_count))
                    
                    if questions:
                        st.session_state.current_questions = questions
                        st.session_state.current_question_index = 0
                        st.session_state.practice_score = {"correct": 0, "total": 0}
                        st.session_state.answer_submitted = False
                        st.success(f"Generated {len(questions)} questions!")
                        st.rerun()
                    else:
                        st.error("Failed to generate questions. Please try again.")
            
            # Current progress
            if st.session_state.current_questions:
                st.subheader("📊 Session Progress")
                
                current_idx = st.session_state.current_question_index
                total_questions = len(st.session_state.current_questions)
                score = st.session_state.practice_score
                
                st.metric("Question", f"{current_idx + 1} / {total_questions}")
                st.metric("Score", f"{score['correct']} / {score['total']}")
                
                if score['total'] > 0:
                    accuracy = (score['correct'] / score['total']) * 100
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Progress bar
                progress = (current_idx + (1 if st.session_state.answer_submitted else 0)) / total_questions
                st.progress(progress)
            
            # Reset session button
            if st.session_state.current_questions:
                st.divider()
                if st.button("🔄 Reset Session", use_container_width=True, key="practice_reset"):
                    st.session_state.current_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.practice_score = {"correct": 0, "total": 0}
                    st.session_state.answer_submitted = False
                    st.rerun()
        
        # Main practice area
        if not st.session_state.current_questions:
            # Welcome screen
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### 🎯 Welcome to Practice Mode!")
                st.markdown("""
                Practice Mode helps you improve your chess skills through progressive difficulty levels:
                
                **📚 Available Levels:**
                - **Level 1**: Piece counting and basic material assessment
                - **Level 2**: Position identification and piece locations
                - **Level 3**: Basic tactical patterns and threats
                - **Level 4**: Strategic analysis and planning
                - **Level 5**: Complex reasoning and advanced concepts
                
                **🚀 How to Start:**
                1. Select your difficulty level in the sidebar
                2. Choose number of questions (1-10)
                3. Click "Start Practice Session"
                4. Answer each question and get instant feedback!
                
                **🏆 Track Your Progress:**
                - See your accuracy and score in real-time
                - Review explanations for each answer
                - Advance to higher levels as you improve
                """)
                
                # Get level information
                level_info = asyncio.run(self.get_question_levels())
                if level_info:
                    with st.expander("📖 Level Details", expanded=False):
                        levels = level_info.get("levels", {})
                        for level_num, level_data in levels.items():
                            st.markdown(f"**{level_data['name']}**: {level_data['description']}")
        
        else:
            # Active practice session
            current_question = st.session_state.current_questions[st.session_state.current_question_index]
            
            # Main content
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("♟️ Chess Position")
                
                # Display chess board
                board_svg = self.render_chess_board(current_question["fen"])
                st.markdown(
                    f'<div class="chess-board">{board_svg}</div>', 
                    unsafe_allow_html=True
                )
                
                # Question info
                st.info(f"**Level {current_question['level']}** - {current_question['question_type'].replace('_', ' ').title()}")
            
            with col2:
                st.subheader("🤔 Question")
                
                # Display question
                st.markdown(f"### {current_question['question_text']}")
                
                if not st.session_state.answer_submitted:
                    # Answer input
                    user_answer = st.text_input(
                        "Your answer:",
                        placeholder="Enter your answer...",
                        key=f"answer_{st.session_state.current_question_index}"
                    )
                    
                    col_submit, col_skip = st.columns([2, 1])
                    
                    with col_submit:
                        submit_button = st.button("✅ Submit Answer", use_container_width=True, 
                                                type="primary", key="practice_submit")
                    
                    with col_skip:
                        skip_button = st.button("⏭️ Skip", use_container_width=True, key="practice_skip")
                    
                    # Handle answer submission
                    if submit_button and user_answer.strip():
                        # Check answer
                        correct_answer = current_question["correct_answer"].lower().strip()
                        user_answer_clean = user_answer.lower().strip()
                        
                        # Check against correct answer and alternatives
                        alternatives = [alt.lower().strip() for alt in current_question.get("alternative_answers", [])]
                        is_correct = (user_answer_clean == correct_answer or 
                                    user_answer_clean in alternatives)
                        
                        # Update score
                        st.session_state.practice_score["total"] += 1
                        if is_correct:
                            st.session_state.practice_score["correct"] += 1
                        
                        # Mark as submitted
                        st.session_state.answer_submitted = True
                        st.session_state.last_answer_correct = is_correct
                        st.rerun()
                    
                    elif skip_button:
                        # Skip question
                        st.session_state.practice_score["total"] += 1
                        st.session_state.answer_submitted = True
                        st.session_state.last_answer_correct = False
                        st.rerun()
                
                else:
                    # Show results
                    is_correct = st.session_state.last_answer_correct
                    
                    if is_correct:
                        st.success("🎉 Correct!")
                    else:
                        st.error("❌ Incorrect")
                    
                    # Show correct answer
                    st.markdown(f"**Correct Answer:** {current_question['correct_answer']}")
                    
                    # Show explanation if available
                    if current_question.get("explanation"):
                        st.markdown(f"**Explanation:** {current_question['explanation']}")
                    
                    # Navigation
                    col_next, col_finish = st.columns([2, 1])
                    
                    is_last_question = st.session_state.current_question_index >= len(st.session_state.current_questions) - 1
                    
                    with col_next:
                        if not is_last_question:
                            if st.button("➡️ Next Question", use_container_width=True, 
                                       type="primary", key="practice_next"):
                                st.session_state.current_question_index += 1
                                st.session_state.answer_submitted = False
                                st.rerun()
                        else:
                            st.success("🏁 Session Complete!")
                    
                    with col_finish:
                        if st.button("🏁 Finish Session", use_container_width=True, key="practice_finish"):
                            # Show final results
                            self.show_practice_results()
                            st.session_state.current_questions = []
                            st.session_state.current_question_index = 0
                            st.session_state.answer_submitted = False
                            st.rerun()
    
    def show_practice_results(self):
        """Show final practice session results."""
        score = st.session_state.practice_score
        accuracy = (score['correct'] / score['total'] * 100) if score['total'] > 0 else 0
        
        st.balloons()
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions Answered", score['total'])
        
        with col2:
            st.metric("Correct Answers", score['correct'])
        
        with col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Performance feedback
        if accuracy >= 80:
            st.success("🌟 Excellent work! You're ready for the next level!")
        elif accuracy >= 60:
            st.info("👍 Good job! Keep practicing to improve further.")
        else:
            st.warning("📚 Keep studying! Try reviewing the fundamentals for this level.")
        
        # Encourage next steps
        current_level = st.session_state.practice_level
        if accuracy >= 80 and current_level < 5:
            st.info(f"💡 Consider trying Level {current_level + 1} for a greater challenge!")
            if st.button(f"🚀 Try Level {current_level + 1}", key="practice_advance_level"):
                st.session_state.practice_level = current_level + 1
                st.rerun()


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