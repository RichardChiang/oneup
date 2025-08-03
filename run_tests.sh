#!/bin/bash

# Run tests with minimal output
echo "Running unit tests..."
python -m pytest tests/backend/test_question_service.py -q --tb=short

echo -e "\nRunning integration tests..."
python -m pytest tests/backend/test_question_api.py -q --tb=short

echo -e "\nRunning all tests with coverage..."
python -m pytest tests/ -q --cov=backend.api.services.question_service --cov-report=term-missing:skip-covered