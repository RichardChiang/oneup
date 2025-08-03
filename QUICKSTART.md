# Chess RL Training System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will get you up and running with the Chess RL Training System quickly.

### Prerequisites

- **Python 3.11+** 
- **Docker & Docker Compose**
- **8GB+ RAM** (recommended)
- **10GB+ free disk space**

### 1. Clone and Setup

```bash
git clone <repository-url>
cd self-learning-rl

# Run the automated setup script
./scripts/setup_dev_environment.sh
```

The setup script will:
- ✅ Install system dependencies (Stockfish, PostgreSQL client)
- ✅ Create Python virtual environments
- ✅ Install all Python packages
- ✅ Start database and Redis services
- ✅ Initialize database tables
- ✅ Create configuration files

### 2. Start the Services

**Terminal 1 - Backend API:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend Interface:**
```bash
./start_frontend.sh
```

### 3. Open the Application

Navigate to: **http://localhost:8501**

🎉 **You're ready to start training!**

---

## 🎯 First Steps

### Load a Chess Puzzle
1. Click **"🎲 Get New Puzzle"** in the sidebar
2. Select your preferred difficulty range
3. A chess position will appear with puzzle information

### Chat with the AI
1. Type a question like: *"What is the best move in this position?"*
2. Click **"🚀 Send Message"**
3. Read the AI's analysis

### Provide Feedback
1. Rate responses with 👍 or 👎 buttons
2. Add optional comments for detailed feedback
3. Your feedback trains the system!

---

## 🛠️ Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_URL=postgresql://chess_user:chess_pass@localhost:5432/chess_rl

# Model (start with a small model)
MODEL_PATH=microsoft/DialoGPT-medium

# Chess Engine
STOCKFISH_PATH=/usr/local/bin/stockfish  # macOS with Homebrew
STOCKFISH_DEPTH=15

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:8501"]
```

### Load Sample Data
```bash
cd backend
source venv/bin/activate
python ../scripts/download_lichess_data.py --max-puzzles 1000
```

---

## 📊 Monitoring

### Health Checks
- **Backend Health:** http://localhost:8000/health
- **API Documentation:** http://localhost:8000/docs
- **Frontend Health:** http://localhost:8501/_stcore/health

### View Statistics
- Check the sidebar in the Streamlit app
- API endpoint: http://localhost:8000/stats

### Database
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U chess_user -d chess_rl

# View tables
\dt

# Check puzzle count
SELECT COUNT(*) FROM tactics_puzzles;

# View recent conversations
SELECT * FROM conversations ORDER BY created_at DESC LIMIT 5;
```

---

## 🔧 Troubleshooting

### Common Issues

**"Model service not available"**
```bash
# Check if model is loading (this may take a few minutes)
docker-compose logs backend

# Try a smaller model in .env
MODEL_PATH=microsoft/DialoGPT-small
```

**"Database connection failed"**
```bash
# Restart database services
docker-compose restart postgres redis

# Check if services are running
docker-compose ps
```

**"Stockfish not found"**
```bash
# macOS
brew install stockfish

# Linux
sudo apt-get install stockfish

# Update STOCKFISH_PATH in .env
```

**"Permission denied on scripts"**
```bash
chmod +x scripts/*.sh
chmod +x start_*.sh
```

### Reset Everything
```bash
# Stop all services
docker-compose down

# Remove data (⚠️ This deletes everything!)
docker-compose down -v

# Start fresh
./scripts/setup_dev_environment.sh
```

---

## 🎪 Advanced Usage

### Custom Model Training
```bash
# Export training data
curl "http://localhost:8000/training/export?min_quality=0.8&limit=1000"

# The system automatically converts conversations to training data
# Use the exported JSON for fine-tuning your own models
```

### Production Deployment
```bash
# Use production docker-compose profile
docker-compose --profile production up -d

# This includes:
# - Nginx reverse proxy
# - Celery workers for background tasks
# - Prometheus + Grafana monitoring
```

### API Integration
```python
import httpx

# Get a random puzzle
async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/puzzles/random")
    puzzle = response.json()
    
    # Chat about the puzzle
    chat_response = await client.post(
        "http://localhost:8000/chat/complete",
        json={
            "message": "Analyze this position",
            "fen": puzzle["fen"],
            "session_id": "my_session"
        }
    )
    
    analysis = chat_response.json()
    print(analysis["content"])
```

---

## 📈 Scaling Up

### Full Lichess Dataset
```bash
# Download complete dataset (⚠️ ~3M puzzles, takes hours)
python scripts/download_lichess_data.py

# This will download ~500MB compressed, ~2GB uncompressed
# Processing takes 1-2 hours on modern hardware
```

### Production Model
```bash
# Use a larger, more capable model
MODEL_PATH=microsoft/DialoGPT-large

# Or use a chess-specific model when available
MODEL_PATH=your-org/chess-trained-model
```

### Hardware Recommendations
- **Development:** 8GB RAM, 4 CPU cores, 20GB storage
- **Small Production:** 16GB RAM, 8 CPU cores, 100GB storage
- **Large Scale:** 32GB+ RAM, GPU acceleration, 500GB+ storage

---

## 🆘 Getting Help

### Check Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Integration Tests
```bash
cd tests
python test_integration.py
```

### Community
- 📖 **Documentation:** See full README.md
- 🐛 **Issues:** Report bugs and feature requests
- 💬 **Discussions:** Share your training results!

---

**Happy Chess Training! ♔♕♖♗♘♙**