#!/bin/bash

# CADlingo Quick Start Script
# This script helps you get started with CADlingo production deployment

set -e

echo "=================================="
echo "CADlingo Production Quick Start"
echo "=================================="
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
else
    echo "✗ Docker is not installed. Please install Docker first."
    echo "  Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if command -v docker-compose &> /dev/null; then
    echo "✓ Docker Compose is installed"
else
    echo "✗ Docker Compose is not installed. Please install it first."
    echo "  Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo ""
echo "Choose deployment option:"
echo "1. Docker (Production - Recommended)"
echo "2. Local Development"
echo "3. Generate Documentation Diagrams"
echo "4. Run Tests"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting CADlingo with Docker..."
        echo ""
        
        # Build and start containers
        docker-compose up -d --build
        
        echo ""
        echo "✓ Services started successfully!"
        echo ""
        echo "Access the application:"
        echo "  • Streamlit UI:  http://localhost:8501"
        echo "  • FastAPI Docs:  http://localhost:8000/docs"
        echo "  • Health Check:  http://localhost:8000/health"
        echo ""
        echo "View logs:"
        echo "  docker-compose logs -f cadlingo-ui"
        echo "  docker-compose logs -f cadlingo-api"
        echo ""
        echo "Stop services:"
        echo "  docker-compose down"
        echo ""
        ;;
    
    2)
        echo ""
        echo "Starting CADlingo locally..."
        echo ""
        
        # Check if virtual environment exists
        if [ ! -d "cad-env" ]; then
            echo "Creating virtual environment..."
            python3 -m venv cad-env
        fi
        
        # Activate virtual environment
        source cad-env/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt -q
        
        echo ""
        echo "Choose service to run:"
        echo "1. Streamlit UI"
        echo "2. FastAPI Backend"
        echo "3. Both (separate terminals)"
        echo ""
        read -p "Enter choice (1-3): " service_choice
        
        case $service_choice in
            1)
                echo "Starting Streamlit UI..."
                streamlit run ui/app.py
                ;;
            2)
                echo "Starting FastAPI backend..."
                python ui/api.py
                ;;
            3)
                echo "Please run these in separate terminals:"
                echo "  Terminal 1: streamlit run ui/app.py"
                echo "  Terminal 2: python ui/api.py"
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac
        ;;
    
    3)
        echo ""
        echo "Generating documentation diagrams..."
        echo ""
        
        # Activate virtual environment if exists
        if [ -d "cad-env" ]; then
            source cad-env/bin/activate
        fi
        
        # Install required packages
        pip install matplotlib numpy -q
        
        # Generate diagrams
        python docs/generate_diagrams.py
        
        echo ""
        echo "✓ Diagrams generated in docs/images/"
        echo ""
        ls -lh docs/images/
        ;;
    
    4)
        echo ""
        echo "Running tests..."
        echo ""
        
        # Activate virtual environment
        if [ -d "cad-env" ]; then
            source cad-env/bin/activate
        fi
        
        # Test imports
        echo "Testing module imports..."
        python -c "from src.improvements.improvement_modules import GeometricValidator, GeometricMetrics, AutomatedEvaluator; print('✓ Improvement modules OK')"
        
        # Test API health
        if [ "$choice" = "1" ]; then
            echo "Testing API health..."
            sleep 3
            curl -f http://localhost:8000/health && echo "✓ API health OK" || echo "✗ API not responding"
        fi
        
        echo ""
        echo "✓ Tests completed"
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
