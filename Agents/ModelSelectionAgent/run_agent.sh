#!/bin/bash

# ModelSelectionAgent Runner Script
# This script provides an easy way to run the ModelSelectionAgent with various configurations

set -e  # Exit on any error

# Default configuration
DEFAULT_PORT=8001
DEFAULT_HOST="0.0.0.0"
DEFAULT_ENV="development"
DEFAULT_WORKERS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
    echo -e "${PURPLE}   ModelSelectionAgent Runner${NC}"
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
    echo "  -m, --mock                Create mock model recommendations for testing"
    echo "      --coral               Enable Coral Protocol integration"
    echo "      --no-coral            Disable Coral Protocol integration"
    echo "      --debug               Enable debug mode"
    echo "      --validate            Validate model database"
    echo "      --benchmark           Run performance benchmarks"
    echo "      --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run with default settings"
    echo "  $0 --port 8001 --dev      # Development mode on port 8001"
    echo "  $0 --container --build    # Build and run in container"
    echo "  $0 --test --validate      # Run tests and validate models"
    echo "  $0 --benchmark            # Run performance benchmarks"
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
    
    # Check required Python packages
    print_status "Checking Python packages..."
    python3 -c "import numpy, sklearn, pandas, fastapi" 2>/dev/null || {
        print_warning "Some required packages are missing. Installing..."
        pip install -e .
    }
    
    print_status "Prerequisites check complete ✓"
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
    
    # Run tests with coverage
    if pytest tests/ -v --cov=main --cov-report=term-missing; then
        print_status "All tests passed ✓"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Function to validate model database
validate_model_database() {
    print_status "Validating model database..."
    
    python3 -c "
import asyncio
from main import ModelSelectionAgent

async def validate():
    agent = ModelSelectionAgent()
    models = agent.model_database
    
    total_models = sum(len(models_list) for models_list in models.values())
    print(f'✓ Total models in database: {total_models}')
    
    for problem_type, model_list in models.items():
        print(f'✓ {problem_type.value}: {len(model_list)} models')
        
        for model in model_list:
            # Validate required fields
            assert model.name, f'Model missing name: {model}'
            assert model.sklearn_class, f'Model missing sklearn_class: {model.name}'
            assert 0 <= model.confidence_score <= 1, f'Invalid confidence score: {model.name}'
            assert model.pros, f'Model missing pros: {model.name}'
            assert model.cons, f'Model missing cons: {model.name}'
    
    print('✓ Model database validation complete')

asyncio.run(validate())
"
    
    print_status "Model database validation complete ✓"
}

# Function to run performance benchmarks
run_benchmarks() {
    print_status "Running performance benchmarks..."
    
    python3 -c "
import asyncio
import time
import json
from main import main

async def benchmark():
    # Test cases for benchmarking
    test_cases = [
        {
            'name': 'Simple Classification',
            'input': {
                'analysis': {
                    'metadata': {'rows': 150, 'columns': 5},
                    'ml_insights': {'problem_type': ['Binary classification']}
                }
            }
        },
        {
            'name': 'Complex Multiclass',
            'input': {
                'analysis': {
                    'metadata': {'rows': 10000, 'columns': 50},
                    'feature_analysis': {'counts': {'numeric': 40, 'categorical': 10}},
                    'ml_insights': {'problem_type': ['Multi-class classification']}
                },
                'preferences': {'skill_level': 'advanced'}
            }
        },
        {
            'name': 'Regression Analysis',
            'input': {
                'analysis': {
                    'metadata': {'rows': 5000, 'columns': 20},
                    'ml_insights': {'problem_type': ['Regression']}
                },
                'preferences': {'interpretability': 'high'}
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f\"Benchmarking: {test_case['name']}\")
        
        # Warm up
        await main(test_case['input'])
        
        # Measure performance
        start_time = time.time()
        result = await main(test_case['input'])
        end_time = time.time()
        
        duration = end_time - start_time
        success = result.get('success', False)
        num_recommendations = len(result.get('recommendations', []))
        
        benchmark_result = {
            'test_case': test_case['name'],
            'duration_ms': round(duration * 1000, 2),
            'success': success,
            'recommendations_count': num_recommendations
        }
        
        results.append(benchmark_result)
        print(f\"  Duration: {benchmark_result['duration_ms']}ms\")
        print(f\"  Success: {success}\")
        print(f\"  Recommendations: {num_recommendations}\")
        print()
    
    # Summary
    avg_duration = sum(r['duration_ms'] for r in results) / len(results)
    print(f\"Average response time: {avg_duration:.2f}ms\")
    
    return results

asyncio.run(benchmark())
"
    
    print_status "Performance benchmarks complete ✓"
}

# Function to build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    docker build -t model-selection-agent:latest .
    print_status "Docker image built successfully ✓"
}

# Function to run in Docker container
run_in_container() {
    print_status "Running in Docker container..."
    
    # Build image if requested
    if [[ "$BUILD_IMAGE" == "true" ]]; then
        build_docker_image
    fi
    
    # Create necessary directories
    mkdir -p ./models
    mkdir -p ./logs
    mkdir -p ./cache
    
    # Run container
    docker run \
        --name model-selection-agent-$(date +%s) \
        --rm \
        -p "${PORT}:${PORT}" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/cache:/app/cache" \
        -e "PORT=${PORT}" \
        -e "APP_ENV=${ENVIRONMENT}" \
        -e "DEBUG=${DEBUG}" \
        -e "CORAL_ENABLED=${CORAL_ENABLED}" \
        -e "MAX_RECOMMENDATIONS=${MAX_RECOMMENDATIONS:-5}" \
        model-selection-agent:latest
}

# Function to create mock recommendations
create_mock_recommendations() {
    print_status "Creating mock recommendations for testing..."
    
    python3 -c "
import asyncio
import json
from main import main

async def create_mocks():
    # Mock test cases
    mock_cases = [
        {
            'name': 'Iris Classification',
            'input': {
                'analysis': {
                    'metadata': {'rows': 150, 'columns': 5},
                    'feature_analysis': {
                        'types': {'numeric': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'categorical': ['species']},
                        'counts': {'numeric': 4, 'categorical': 1}
                    },
                    'ml_insights': {'problem_type': ['Multi-class classification (target: species)']}
                },
                'preferences': {'skill_level': 'beginner'}
            }
        },
        {
            'name': 'Boston Housing Regression',
            'input': {
                'analysis': {
                    'metadata': {'rows': 506, 'columns': 14},
                    'feature_analysis': {'counts': {'numeric': 13, 'categorical': 1}},
                    'ml_insights': {'problem_type': ['Regression (target: price)']}
                },
                'preferences': {'skill_level': 'intermediate', 'interpretability': 'high'}
            }
        }
    ]
    
    print('Mock Recommendations:')
    print('=' * 50)
    
    for case in mock_cases:
        print(f\"\\n{case['name']}:\")
        result = await main(case['input'])
        
        if result.get('success'):
            top_rec = result['recommendations'][0]
            print(f\"  Top Model: {top_rec['name']}\")
            print(f\"  Confidence: {top_rec['confidence_score']:.1%}\")
            print(f\"  Complexity: {top_rec['complexity_level']}\")
            print(f\"  Summary: {result['summary'][:100]}...\")
        else:
            print(f\"  Error: {result.get('error', 'Unknown error')}\")

asyncio.run(create_mocks())
"
    
    print_status "Mock recommendations created ✓"
}

# Function to start the agent
start_agent() {
    print_status "Starting ModelSelectionAgent..."
    
    # Prepare command
    if [[ "$DEVELOPMENT" == "true" ]]; then
        cmd="uvicorn main:app --host $HOST --port $PORT --reload --log-level debug"
    else
        cmd="uvicorn main:app --host $HOST --port $PORT --workers $WORKERS --log-level info"
    fi
    
    # Set environment variables
    export PORT="$PORT"
    export HOST="$HOST"
    export APP_ENV="$ENVIRONMENT"
    export DEBUG="$DEBUG"
    export CORAL_ENABLED="$CORAL_ENABLED"
    export MAX_RECOMMENDATIONS="${MAX_RECOMMENDATIONS:-5}"
    
    print_status "Agent configuration:"
    print_status "  → URL: http://${HOST}:${PORT}"
    print_status "  → Health: http://${HOST}:${PORT}/health"
    print_status "  → Docs: http://${HOST}:${PORT}/docs"
    print_status "  → Environment: ${ENVIRONMENT}"
    print_status "  → Coral: ${CORAL_ENABLED}"
    print_status "  → Debug: ${DEBUG}"
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
VALIDATE_DB="false"
RUN_BENCHMARK="false"
MAX_RECOMMENDATIONS=5

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
        --validate)
            VALIDATE_DB="true"
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK="true"
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
    
    # Validate model database if requested
    if [[ "$VALIDATE_DB" == "true" ]]; then
        validate_model_database
    fi
    
    # Create mock data if requested
    if [[ "$CREATE_MOCK" == "true" ]]; then
        create_mock_recommendations
        return 0
    fi
    
    # Run benchmarks if requested
    if [[ "$RUN_BENCHMARK" == "true" ]]; then
        run_benchmarks
        return 0
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
    print_status "Shutting down ModelSelectionAgent..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"