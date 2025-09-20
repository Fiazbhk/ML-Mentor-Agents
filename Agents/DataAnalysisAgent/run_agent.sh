#!/bin/bash

# DataAnalysisAgent Runner Script
# This script provides an easy way to run the DataAnalysisAgent with various configurations

set -e  # Exit on any error

# Default configuration
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"
DEFAULT_ENV="development"
DEFAULT_WORKERS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}   DataAnalysisAgent Runner${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT           Set the port (default: $DEFAULT_PORT)"
    echo "  -h, --host HOST           Set the host (default: $DEFAULT_HOST)"
    echo "  -e, --env ENVIRONMENT     Set environment (development/staging/production)"
    echo "  -w, --workers WORKERS     Number of worker processes (default: $DEFAULT_WORKERS)"
    echo "  -d, --dev                 Run in development mode with auto-reload"
    echo "  -t, --test                Run tests before starting"
    echo "  -c, --container           Run in Docker container"
    echo "  -b, --build               Build Docker image before running"
    echo "  -s, --setup               Setup environment and install dependencies"
    echo "  -m, --mock                Run with mock data for testing"
    echo "      --coral               Enable Coral Protocol integration"
    echo "      --no-coral            Disable Coral Protocol integration"
    echo "      --debug               Enable debug mode"
    echo "      --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run with default settings"
    echo "  $0 --port 8080 --dev      # Development mode on port 8080"
    echo "  $0 --container --build    # Build and run in container"
    echo "  $0 --test --env staging   # Run tests then start in staging mode"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "Python 3.8+ is required, found $python_version"
        exit 1
    fi
    
    print_status "Python $python_version detected ✓"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -e .
    
    # Install development dependencies if in dev mode
    if [[ "$ENVIRONMENT" == "development" ]]; then
        pip install -e .[dev]
    fi
    
    # Copy environment file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        print_status "Creating .env file from template..."
        cp .env_sample .env
        print_warning "Please edit .env file with your configuration"
    fi
    
    print_status "Environment setup complete ✓"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_warning "pytest not found, installing..."
        pip install pytest pytest-asyncio pytest-cov
    fi
    
    # Run tests
    if pytest tests/ -v; then
        print_status "All tests passed ✓"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Function to build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    docker build -t data-analysis-agent:latest .
    print_status "Docker image built successfully ✓"
}

# Function to run in Docker container
run_in_container() {
    print_status "Running in Docker container..."
    
    # Build image if requested
    if [[ "$BUILD_IMAGE" == "true" ]]; then
        build_docker_image
    fi
    
    # Create data directory if it doesn't exist
    mkdir -p ./data
    mkdir -p ./logs
    
    # Run container
    docker run \
        --name data-analysis-agent-$(date +%s) \
        --rm \
        -p "${PORT}:${PORT}" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -e "PORT=${PORT}" \
        -e "APP_ENV=${ENVIRONMENT}" \
        -e "DEBUG=${DEBUG}" \
        -e "CORAL_ENABLED=${CORAL_ENABLED}" \
        data-analysis-agent:latest
}

# Function to create mock data
create_mock_data() {
    print_status "Creating mock data for testing..."
    
    mkdir -p ./data
    
    # Create iris dataset if it doesn't exist
    if [[ ! -f "./data/iris.csv" ]]; then
        cat > ./data/iris.csv << 'EOF'
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
EOF
        print_status "Created iris.csv sample dataset"
    fi
    
    # Create housing dataset if it doesn't exist
    if [[ ! -f "./data/housing.csv" ]]; then
        cat > ./data/housing.csv << 'EOF'
size,bedrooms,bathrooms,age,price
1200,2,1,10,150000
1500,3,2,5,200000
1800,3,2,15,180000
2000,4,3,2,250000
2200,4,3,8,280000
1000,2,1,20,120000
1300,2,2,12,160000
1600,3,2,7,190000
1900,3,2,18,175000
2100,4,3,4,240000
EOF
        print_status "Created housing.csv sample dataset"
    fi
    
    print_status "Mock data ready ✓"
}

# Function to start the agent
start_agent() {
    print_status "Starting DataAnalysisAgent..."
    
    # Prepare command
    if [[ "$DEVELOPMENT" == "true" ]]; then
        cmd="uvicorn main:app --host $HOST --port $PORT --reload"
    else
        cmd="uvicorn main:app --host $HOST --port $PORT --workers $WORKERS"
    fi
    
    # Add environment variables
    export PORT="$PORT"
    export HOST="$HOST"
    export APP_ENV="$ENVIRONMENT"
    export DEBUG="$DEBUG"
    export CORAL_ENABLED="$CORAL_ENABLED"
    
    print_status "Agent will be available at: http://${HOST}:${PORT}"
    print_status "Health check: http://${HOST}:${PORT}/health"
    print_status "API docs: http://${HOST}:${PORT}/docs"
    print_status ""
    print_status "Press Ctrl+C to stop the agent"
    print_status ""
    
    # Start the agent
    eval $cmd
}

# Parse command line arguments
PORT=$DEFAULT_PORT
HOST=$DEFAULT_HOST
ENVIRONMENT=$DEFAULT_ENV
WORKERS=$DEFAULT_WORKERS
DEVELOPMENT="false"
RUN_TESTS="false"
USE_CONTAINER="false"
BUILD_IMAGE="false"
SETUP_ENV="false"
CREATE_MOCK="false"
DEBUG="false"
CORAL_ENABLED="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -d|--dev)
            DEVELOPMENT="true"
            ENVIRONMENT="development"
            DEBUG="true"
            shift
            ;;
        -t|--test)
            RUN_TESTS="true"
            shift
            ;;
        -c|--container)
            USE_CONTAINER="true"
            shift
            ;;
        -b|--build)
            BUILD_IMAGE="true"
            shift
            ;;
        -s|--setup)
            SETUP_ENV="true"
            shift
            ;;
        -m|--mock)
            CREATE_MOCK="true"
            shift
            ;;
        --coral)
            CORAL_ENABLED="true"
            shift
            ;;
        --no-coral)
            CORAL_ENABLED="false"
            shift
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    # Setup environment if requested
    if [[ "$SETUP_ENV" == "true" ]]; then
        check_prerequisites
        setup_environment
    fi
    
    # Create mock data if requested
    if [[ "$CREATE_MOCK" == "true" ]]; then
        create_mock_data
    fi
    
    # Run tests if requested
    if [[ "$RUN_TESTS" == "true" ]]; then
        run_tests
    fi
    
    # Run in container or locally
    if [[ "$USE_CONTAINER" == "true" ]]; then
        run_in_container
    else
        # Check prerequisites for local run
        if [[ "$SETUP_ENV" != "true" ]]; then
            check_prerequisites
        fi
        
        # Activate virtual environment if it exists
        if [[ -d "venv" ]]; then
            source venv/bin/activate
        fi
        
        # Start the agent
        start_agent
    fi
}

# Trap Ctrl+C and cleanup
cleanup() {
    print_status "Shutting down agent..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"
        