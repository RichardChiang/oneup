#!/bin/bash

# Chess RL Training System - Development Environment Setup
# This script sets up the complete development environment

set -e  # Exit on any error

echo "♔ Chess RL Training System - Development Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# 1. Check Prerequisites
print_status "Checking prerequisites..."

# Check Python 3.11+
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3.11+ is required but not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker found"
else
    print_error "Docker is required but not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    print_success "Docker Compose found"
else
    print_error "Docker Compose is required but not found"
    exit 1
fi

# 2. Install System Dependencies
print_status "Installing system dependencies..."

if [[ "$OS" == "macos" ]]; then
    # macOS with Homebrew
    if command -v brew &> /dev/null; then
        print_status "Installing Stockfish via Homebrew..."
        brew install stockfish || print_warning "Stockfish installation failed"
        
        print_status "Installing PostgreSQL client..."
        brew install postgresql || print_warning "PostgreSQL client installation failed"
    else
        print_warning "Homebrew not found. Please install Stockfish and PostgreSQL manually."
    fi
elif [[ "$OS" == "linux" ]]; then
    # Linux with apt
    if command -v apt-get &> /dev/null; then
        print_status "Installing system packages..."
        sudo apt-get update
        sudo apt-get install -y stockfish postgresql-client python3-venv python3-pip || print_warning "Some packages failed to install"
    else
        print_warning "apt-get not found. Please install Stockfish and PostgreSQL client manually."
    fi
fi

# 3. Create Python Virtual Environment
print_status "Setting up Python virtual environments..."

# Backend environment
if [ ! -d "backend/venv" ]; then
    print_status "Creating backend virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    cd ..
    print_success "Backend environment created"
else
    print_success "Backend environment already exists"
fi

# Frontend environment
if [ ! -d "frontend/venv" ]; then
    print_status "Creating frontend virtual environment..."
    cd frontend
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    cd ..
    print_success "Frontend environment created"
else
    print_success "Frontend environment already exists"
fi

# 4. Set up Environment Configuration
print_status "Setting up environment configuration..."

if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
    print_warning "Please review and update .env file with your specific configuration"
else
    print_success ".env file already exists"
fi

# 5. Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/lichess
mkdir -p logs
mkdir -p models
mkdir -p checkpoints
print_success "Directories created"

# 6. Start Development Services
print_status "Starting development services..."

# Start database and redis
docker-compose up -d postgres redis

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check if PostgreSQL is ready
if docker-compose exec postgres pg_isready -U chess_user -d chess_rl; then
    print_success "PostgreSQL is ready"
else
    print_error "PostgreSQL failed to start"
    exit 1
fi

# Check if Redis is ready
if docker-compose exec redis redis-cli ping | grep -q PONG; then
    print_success "Redis is ready"
else
    print_error "Redis failed to start"
    exit 1
fi

# 7. Initialize Database
print_status "Initializing database..."

cd backend
source venv/bin/activate
python -c "
from database import initialize_database
import os
db = initialize_database(os.getenv('DATABASE_URL', 'postgresql://chess_user:chess_pass@localhost:5432/chess_rl'))
db.create_tables()
print('Database tables created successfully')
"
deactivate
cd ..

print_success "Database initialized"

# 8. Download Sample Data (optional)
read -p "Would you like to download sample Lichess puzzle data? This may take several minutes. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading sample puzzle data..."
    cd backend
    source venv/bin/activate
    python ../scripts/download_lichess_data.py --max-puzzles 1000
    deactivate
    cd ..
    print_success "Sample data downloaded"
else
    print_status "Skipping sample data download"
fi

# 9. Test Installation
print_status "Testing installation..."

# Test backend
print_status "Testing backend services..."
cd backend
source venv/bin/activate

# Test chess converter
python -c "
from chess import ChessConverter
result = ChessConverter.fen_to_unicode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
print(f'Chess converter test: {len(result) == 64}')
"

# Test database connection
python -c "
import asyncio
from database import initialize_database, get_database
import os

async def test_db():
    try:
        db = initialize_database(os.getenv('DATABASE_URL', 'postgresql://chess_user:chess_pass@localhost:5432/chess_rl'))
        connected = await db.check_connection()
        print(f'Database connection test: {connected}')
        await db.close()
    except Exception as e:
        print(f'Database test failed: {e}')

asyncio.run(test_db())
"

deactivate
cd ..

print_success "Installation tests completed"

# 10. Create Start Scripts
print_status "Creating start scripts..."

# Backend start script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
EOF

# Frontend start script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
EOF

# Make scripts executable
chmod +x start_backend.sh start_frontend.sh

print_success "Start scripts created"

# 11. Display Setup Summary
echo
echo "🎉 Development Environment Setup Complete!"
echo "=========================================="
echo
echo "📋 Quick Start Commands:"
echo "  • Start all services:     docker-compose up -d"
echo "  • Start backend:          ./start_backend.sh"
echo "  • Start frontend:         ./start_frontend.sh"
echo "  • Stop services:          docker-compose down"
echo "  • View logs:              docker-compose logs -f"
echo
echo "🌐 URLs:"
echo "  • Frontend (Streamlit):   http://localhost:8501"
echo "  • Backend API:            http://localhost:8000"
echo "  • API Documentation:      http://localhost:8000/docs"
echo "  • Database:               postgresql://chess_user:chess_pass@localhost:5432/chess_rl"
echo
echo "📁 Key Files:"
echo "  • Environment config:     .env"
echo "  • Docker services:        docker-compose.yml"
echo "  • Backend code:           backend/"
echo "  • Frontend code:          frontend/"
echo "  • Data directory:         data/"
echo
echo "🔧 Next Steps:"
echo "  1. Review and update .env file if needed"
echo "  2. Start the backend: ./start_backend.sh"
echo "  3. In another terminal, start frontend: ./start_frontend.sh"
echo "  4. Open http://localhost:8501 in your browser"
echo "  5. Start chatting with the chess AI!"
echo
echo "📚 For more information, see README.md"
echo

print_success "Setup completed successfully! 🚀"