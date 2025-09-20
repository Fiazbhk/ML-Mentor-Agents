#!/bin/bash

# ML Mentor Agents Dependency Check Script
# This script verifies all required dependencies for DataAnalysisAgent and ModelSelectionAgent
# and optionally installs missing components.

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# Ensure that pipeline commands fail on the first error.
set -o pipefail

# --- Color Definitions ---
C_BLUE='\033[0;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_RED='\033[0;31m'
C_PURPLE='\033[0;35m'
C_CYAN='\033[0;36m'
C_NC='\033[0m' # No Color

# --- Global Variables ---
MISSING_DEPS=0
PYTHON_PACKAGES_MISSING=0
SYSTEM_DEPS_MISSING=0

# --- Utility Functions ---

# Function to print colored headers
print_header() {
    echo -e "${C_BLUE}=====================================${C_NC}"
    echo -e "${C_PURPLE}   ML Mentor Agents Dependency Check${C_NC}"
    echo -e "${C_BLUE}=====================================${C_NC}\n"
}

print_section() {
    echo -e "${C_CYAN}$1${C_NC}"
    echo -e "${C_CYAN}$(printf '%.0s-' {1..50})${C_NC}"
}

# Function to compare version numbers
version_compare() {
    local version1=$1
    local version2=$2

    # Convert versions to comparable format (remove non-numeric characters except dots)
    version1=$(echo "$version1" | sed 's/[^0-9.]//g')
    version2=$(echo "$version2" | sed 's/[^0-9.]//g')

    # Cross-platform version comparison
    if printf '%s\n%s\n' "$version2" "$version1" | sort -V -C 2>/dev/null; then
        return 0  # version1 >= version2
    elif command -v sort >/dev/null 2>&1 && sort --version-sort /dev/null 2>/dev/null; then
        if printf '%s\n%s\n' "$version2" "$version1" | sort --version-sort -C; then
            return 0
        else
            return 1
        fi
    else
        # Fallback: manual version comparison
        IFS='.' read -ra ver1_parts <<< "$version1"
        IFS='.' read -ra ver2_parts <<< "$version2"

        local max_len=${#ver1_parts[@]}
        if [ ${#ver2_parts[@]} -gt $max_len ]; then
            max_len=${#ver2_parts[@]}
        fi

        for ((i=0; i<max_len; i++)); do
            local v1=${ver1_parts[i]:-0}
            local v2=${ver2_parts[i]:-0}

            if [ "$v1" -gt "$v2" ]; then
                return 0
            elif [ "$v1" -lt "$v2" ]; then
                return 1
            fi
        done

        return 0
    fi
}

# --- System Dependencies Check ---

# Function to check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${C_RED}  [✗] Python 3 is not installed.${C_NC}"
        return 1
    fi

    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local required_version="3.8"

    if version_compare "$python_version" "$required_version"; then
        echo -e "${C_GREEN}  [✓] Python $python_version (required: $required_version+)${C_NC}"
        return 0
    else
        echo -e "${C_RED}  [✗] Python $python_version (required: $required_version+)${C_NC}"
        return 1
    fi
}

# Function to check pip
check_pip() {
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        echo -e "${C_RED}  [✗] pip is not installed.${C_NC}"
        return 1
    fi

    local pip_version
    if command -v pip3 &> /dev/null; then
        pip_version=$(pip3 --version 2>&1 | cut -d' ' -f2)
    else
        pip_version=$(python3 -m pip --version 2>&1 | cut -d' ' -f2)
    fi

    echo -e "${C_GREEN}  [✓] pip $pip_version${C_NC}"
    return 0
}

# Function to check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${C_YELLOW}  [!] Docker is not installed (optional for containerized deployment).${C_NC}"
        return 0
    fi

    local docker_version=$(docker --version 2>&1 | cut -d' ' -f3 | sed 's/,//')
    echo -e "${C_GREEN}  [✓] Docker $docker_version (optional)${C_NC}"
    return 0
}

# Function to check Git
check_git() {
    if ! command -v git &> /dev/null; then
        echo -e "${C_RED}  [✗] Git is not installed.${C_NC}"
        return 1
    fi

    local git_version=$(git --version 2>&1 | cut -d' ' -f3)
    echo -e "${C_GREEN}  [✓] Git $git_version${C_NC}"
    return 0
}

# Function to check curl
check_curl() {
    if ! command -v curl &> /dev/null; then
        echo -e "${C_RED}  [✗] curl is not installed.${C_NC}"
        return 1
    fi

    local curl_version=$(curl --version 2>&1 | head -n1 | cut -d' ' -f2)
    echo -e "${C_GREEN}  [✓] curl $curl_version${C_NC}"
    return 0
}

# Function to check jq (for JSON processing)
check_jq() {
    if ! command -v jq &> /dev/null; then
        echo -e "${C_YELLOW}  [!] jq is not installed (recommended for JSON processing).${C_NC}"
        return 0
    fi

    local jq_version=$(jq --version 2>&1 | sed 's/jq-//')
    echo -e "${C_GREEN}  [✓] jq $jq_version (recommended)${C_NC}"
    return 0
}

# --- Python Package Dependencies Check ---

# Function to check if a Python package is installed
check_python_package() {
    local package_name=$1
    local required_version=${2:-""}
    local display_name=${3:-$package_name}

    if python3 -c "import $package_name" &> /dev/null; then
        local installed_version=""
        
        # Try to get version
        if [ -n "$required_version" ]; then
            installed_version=$(python3 -c "import $package_name; print(getattr($package_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            
            if [ "$installed_version" != "unknown" ] && version_compare "$installed_version" "$required_version"; then
                echo -e "${C_GREEN}  [✓] $display_name $installed_version (required: $required_version+)${C_NC}"
                return 0
            elif [ "$installed_version" != "unknown" ]; then
                echo -e "${C_RED}  [✗] $display_name $installed_version (required: $required_version+)${C_NC}"
                return 1
            else
                echo -e "${C_GREEN}  [✓] $display_name (version unknown)${C_NC}"
                return 0
            fi
        else
            echo -e "${C_GREEN}  [✓] $display_name${C_NC}"
            return 0
        fi
    else
        echo -e "${C_RED}  [✗] $display_name is not installed${C_NC}"
        return 1
    fi
}

# Function to check core ML packages
check_ml_packages() {
    print_section "Machine Learning Core Packages"
    
    local ml_missing=0
    
    # Core scientific computing
    check_python_package "numpy" "1.20.0" "NumPy" || ml_missing=1
    check_python_package "pandas" "1.3.0" "Pandas" || ml_missing=1
    check_python_package "scipy" "1.7.0" "SciPy" || ml_missing=1
    
    # Machine learning
    check_python_package "sklearn" "1.0.0" "scikit-learn" || ml_missing=1
    
    # Optional ML packages
    echo -e "\n${C_CYAN}Optional ML Enhancement Packages:${C_NC}"
    check_python_package "xgboost" "" "XGBoost (optional)" || true
    check_python_package "lightgbm" "" "LightGBM (optional)" || true
    check_python_package "catboost" "" "CatBoost (optional)" || true
    
    return $ml_missing
}

# Function to check web framework packages
check_web_packages() {
    print_section "Web Framework Packages"
    
    local web_missing=0
    
    check_python_package "fastapi" "0.68.0" "FastAPI" || web_missing=1
    check_python_package "uvicorn" "0.15.0" "Uvicorn" || web_missing=1
    check_python_package "pydantic" "1.8.0" "Pydantic" || web_missing=1
    
    return $web_missing
}

# Function to check utility packages
check_utility_packages() {
    print_section "Utility Packages"
    
    local util_missing=0
    
    check_python_package "dotenv" "" "python-dotenv" || util_missing=1
    check_python_package "aiofiles" "" "aiofiles" || util_missing=1
    check_python_package "loguru" "" "loguru" || util_missing=1
    check_python_package "openpyxl" "" "openpyxl" || util_missing=1
    
    return $util_missing
}

# Function to check development packages
check_dev_packages() {
    print_section "Development Packages (Optional)"
    
    echo -e "${C_CYAN}These packages are needed for development and testing:${C_NC}"
    
    check_python_package "pytest" "" "pytest (testing)" || true
    check_python_package "black" "" "black (code formatting)" || true
    check_python_package "isort" "" "isort (import sorting)" || true
    check_python_package "flake8" "" "flake8 (linting)" || true
    check_python_package "mypy" "" "mypy (type checking)" || true
    
    return 0
}

# Function to check visualization packages
check_viz_packages() {
    print_section "Visualization Packages (Optional)"
    
    echo -e "${C_CYAN}These packages enhance data visualization capabilities:${C_NC}"
    
    check_python_package "matplotlib" "" "Matplotlib (plotting)" || true
    check_python_package "seaborn" "" "Seaborn (statistical plots)" || true
    check_python_package "plotly" "" "Plotly (interactive plots)" || true
    
    return 0
}

# --- Environment Configuration Check ---

check_environment_config() {
    print_section "Environment Configuration"
    
    local config_issues=0
    
    # Check for .env files
    if [ -f ".env" ]; then
        echo -e "${C_GREEN}  [✓] .env file found${C_NC}"
    else
        if [ -f ".env_sample" ]; then
            echo -e "${C_YELLOW}  [!] .env file missing, but .env_sample found${C_NC}"
            echo -e "${C_YELLOW}      Run: cp .env_sample .env${C_NC}"
        else
            echo -e "${C_RED}  [✗] No .env or .env_sample file found${C_NC}"
            config_issues=1
        fi
    fi
    
    # Check for required directories
    local required_dirs=("logs" "data" "models")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${C_GREEN}  [✓] Directory '$dir' exists${C_NC}"
        else
            echo -e "${C_YELLOW}  [!] Directory '$dir' will be created automatically${C_NC}"
        fi
    done
    
    # Check for agent files
    if [ -f "main.py" ]; then
        echo -e "${C_GREEN}  [✓] main.py found${C_NC}"
    else
        echo -e "${C_RED}  [✗] main.py not found${C_NC}"
        config_issues=1
    fi
    
    if [ -f "coral-agent.toml" ]; then
        echo -e "${C_GREEN}  [✓] coral-agent.toml found${C_NC}"
    else
        echo -e "${C_YELLOW}  [!] coral-agent.toml not found (optional)${C_NC}"
    fi
    
    return $config_issues
}

# --- Port Availability Check ---

check_port_availability() {
    print_section "Port Availability Check"
    
    local ports=(8000 8001 8002 8003)
    local port_issues=0
    
    for port in "${ports[@]}"; do
        if command -v lsof &> /dev/null; then
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                echo -e "${C_YELLOW}  [!] Port $port is in use${C_NC}"
            else
                echo -e "${C_GREEN}  [✓] Port $port is available${C_NC}"
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -tuln 2>/dev/null | grep ":$port " >/dev/null; then
                echo -e "${C_YELLOW}  [!] Port $port is in use${C_NC}"
            else
                echo -e "${C_GREEN}  [✓] Port $port is available${C_NC}"
            fi
        else
            echo -e "${C_YELLOW}  [!] Cannot check port $port (lsof/netstat not available)${C_NC}"
        fi
    done
    
    return $port_issues
}

# --- Installation Helper Functions ---

install_python_packages() {
    echo -e "\n${C_BLUE}Would you like to install missing Python packages? (y/n): ${C_NC}"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${C_YELLOW}Installing Python packages...${C_NC}"
        
        # Install core packages
        echo -e "${C_CYAN}Installing core packages...${C_NC}"
        python3 -m pip install --upgrade pip
        python3 -m pip install \
            "numpy>=1.20.0" \
            "pandas>=1.3.0" \
            "scipy>=1.7.0" \
            "scikit-learn>=1.0.0" \
            "fastapi>=0.68.0" \
            "uvicorn>=0.15.0" \
            "pydantic>=1.8.0" \
            "python-dotenv>=0.19.0" \
            "aiofiles>=0.7.0" \
            "loguru>=0.5.0" \
            "openpyxl>=3.0.0" \
            "python-multipart>=0.0.5"
        
        echo -e "${C_GREEN}✅ Core packages installed successfully!${C_NC}"
        
        # Ask about optional packages
        echo -e "\n${C_BLUE}Install optional enhancement packages (visualization, development tools)? (y/n): ${C_NC}"
        read -r opt_response
        
        if [[ "$opt_response" =~ ^[Yy]$ ]]; then
            echo -e "${C_CYAN}Installing optional packages...${C_NC}"
            python3 -m pip install \
                "matplotlib>=3.4.0" \
                "seaborn>=0.11.0" \
                "plotly>=5.0.0" \
                "pytest>=6.2.0" \
                "black>=21.0.0" \
                "isort>=5.9.0" \
                "flake8>=3.9.0"
            
            echo -e "${C_GREEN}✅ Optional packages installed successfully!${C_NC}"
        fi
    fi
}

create_env_file() {
    if [ -f ".env_sample" ] && [ ! -f ".env" ]; then
        echo -e "\n${C_BLUE}Would you like to create .env file from template? (y/n): ${C_NC}"
        read -r response

        if [[ "$response" =~ ^[Yy]$ ]]; then
            cp .env_sample .env
            echo -e "${C_GREEN}✅ .env file created from template${C_NC}"
            echo -e "${C_YELLOW}⚠️  Please edit .env file with your specific configuration${C_NC}"
        fi
    fi
}

create_directories() {
    echo -e "\n${C_BLUE}Creating required directories...${C_NC}"
    
    local dirs=("logs" "data" "models" "cache" "temp")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${C_GREEN}  [✓] Created directory: $dir${C_NC}"
        fi
    done
}

# --- Main Dependency Check Function ---

check_system_dependencies() {
    print_section "System Dependencies"
    
    local sys_missing=0
    
    check_python || sys_missing=1
    check_pip || sys_missing=1
    check_git || sys_missing=1
    check_curl || sys_missing=1
    check_jq
    check_docker
    
    SYSTEM_DEPS_MISSING=$sys_missing
    return $sys_missing
}

check_python_dependencies() {
    local py_missing=0
    
    check_ml_packages || py_missing=1
    echo ""
    check_web_packages || py_missing=1
    echo ""
    check_utility_packages || py_missing=1
    echo ""
    check_dev_packages
    echo ""
    check_viz_packages
    
    PYTHON_PACKAGES_MISSING=$py_missing
    return $py_missing
}

# --- Agent-Specific Checks ---

check_agent_specific() {
    print_section "ML Mentor Agent Specific Checks"
    
    local agent_issues=0
    
    # Check if we're in the right directory
    if [ -f "main.py" ] && grep -q "DataAnalysisAgent\|ModelSelectionAgent\|TrainingAgent" main.py 2>/dev/null; then
        echo -e "${C_GREEN}  [✓] ML Mentor agent detected${C_NC}"
        
        # Determine agent type
        if grep -q "DataAnalysisAgent" main.py; then
            echo -e "${C_GREEN}  [✓] DataAnalysisAgent detected${C_NC}"
        fi
        if grep -q "ModelSelectionAgent" main.py; then
            echo -e "${C_GREEN}  [✓] ModelSelectionAgent detected${C_NC}"
        fi
        if grep -q "TrainingAgent" main.py; then
            echo -e "${C_GREEN}  [✓] TrainingAgent detected${C_NC}"
        fi
    else
        echo -e "${C_YELLOW}  [!] ML Mentor agent files not found in current directory${C_NC}"
    fi
    
    # Check for pyproject.toml
    if [ -f "pyproject.toml" ]; then
        echo -e "${C_GREEN}  [✓] pyproject.toml found${C_NC}"
    else
        echo -e "${C_YELLOW}  [!] pyproject.toml not found (recommended for package management)${C_NC}"
    fi
    
    # Check for run script
    if [ -f "run_agent.sh" ]; then
        echo -e "${C_GREEN}  [✓] run_agent.sh found${C_NC}"
        if [ -x "run_agent.sh" ]; then
            echo -e "${C_GREEN}  [✓] run_agent.sh is executable${C_NC}"
        else
            echo -e "${C_YELLOW}  [!] run_agent.sh is not executable (run: chmod +x run_agent.sh)${C_NC}"
        fi
    else
        echo -e "${C_YELLOW}  [!] run_agent.sh not found${C_NC}"
    fi
    
    return $agent_issues
}

# --- Summary and Recommendations ---

print_summary() {
    echo -e "\n${C_BLUE}=====================================${C_NC}"
    echo -e "${C_BLUE}           DEPENDENCY SUMMARY        ${C_NC}"
    echo -e "${C_BLUE}=====================================${C_NC}\n"
    
    if [ $SYSTEM_DEPS_MISSING -eq 0 ] && [ $PYTHON_PACKAGES_MISSING -eq 0 ]; then
        echo -e "${C_GREEN}🎉 All required dependencies are satisfied!${C_NC}\n"
        
        echo -e "${C_BLUE}✅ Your ML Mentor agents are ready to run!${C_NC}\n"
        
        echo -e "${C_CYAN}Quick Start Commands:${C_NC}"
        echo -e "  ${C_YELLOW}# DataAnalysisAgent (port 8000)${C_NC}"
        echo -e "  ./run_agent.sh --dev --port 8000"
        echo -e ""
        echo -e "  ${C_YELLOW}# ModelSelectionAgent (port 8001)${C_NC}"
        echo -e "  ./run_agent.sh --dev --port 8001"
        echo -e ""
        echo -e "  ${C_YELLOW}# TrainingAgent (port 8002)${C_NC}"
        echo -e "  ./run_agent.sh --dev --port 8002"
        echo -e ""
        echo -e "  ${C_YELLOW}# Test with mock data${C_NC}"
        echo -e "  ./run_agent.sh --mock"
        echo -e ""
        echo -e "  ${C_YELLOW}# Run in Docker${C_NC}"
        echo -e "  ./run_agent.sh --container --build"
        
    else
        echo -e "${C_RED}❌ Some dependencies are missing or need attention.${C_NC}\n"
        
        if [ $SYSTEM_DEPS_MISSING -eq 1 ]; then
            echo -e "${C_RED}System Dependencies Issues:${C_NC}"
            echo -e "  - Install missing system packages using your package manager"
            echo -e "  - Ensure Python 3.8+ is installed"
            echo -e ""
        fi
        
        if [ $PYTHON_PACKAGES_MISSING -eq 1 ]; then
            echo -e "${C_RED}Python Package Issues:${C_NC}"
            echo -e "  - Run this script with --install to auto-install packages"
            echo -e "  - Or manually install: ${C_YELLOW}pip install -e .${C_NC}"
            echo -e ""
        fi
        
        echo -e "${C_BLUE}Installation Commands:${C_NC}"
        echo -e "  ${C_YELLOW}# Auto-install Python packages${C_NC}"
        echo -e "  $0 --install"
        echo -e ""
        echo -e "  ${C_YELLOW}# Manual installation${C_NC}"
        echo -e "  pip install -e ."
        echo -e ""
        echo -e "  ${C_YELLOW}# Create environment file${C_NC}"
        echo -e "  cp .env_sample .env"
    fi
    
    echo -e "\n${C_CYAN}Useful Resources:${C_NC}"
    echo -e "  📖 README.md - Complete documentation"
    echo -e "  🔧 .env_sample - Environment configuration template"
    echo -e "  🐳 Dockerfile - Container deployment"
    echo -e "  🚀 run_agent.sh - Agent runner script"
    echo -e ""
    echo -e "${C_BLUE}For help: ./run_agent.sh --help${C_NC}"
}

# --- Main Function ---

main() {
    print_header
    
    # Parse command line arguments
    INSTALL_PACKAGES=false
    
    for arg in "$@"; do
        case $arg in
            --install)
                INSTALL_PACKAGES=true
                shift
                ;;
            --help|-h)
                echo "ML Mentor Agents Dependency Check"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --install    Automatically install missing Python packages"
                echo "  --help, -h   Show this help message"
                echo ""
                echo "This script checks all dependencies for ML Mentor agents:"
                echo "  - DataAnalysisAgent"
                echo "  - ModelSelectionAgent" 
                echo "  - TrainingAgent"
                echo ""
                exit 0
                ;;
        esac
    done
    
    # Run all checks
    check_system_dependencies
    echo ""
    check_python_dependencies
    echo ""
    check_environment_config
    echo ""
    check_port_availability
    echo ""
    check_agent_specific
    
    # Install packages if requested
    if [ "$INSTALL_PACKAGES" = true ]; then
        if [ $PYTHON_PACKAGES_MISSING -eq 1 ]; then
            install_python_packages
        fi
        create_env_file
        create_directories
    elif [ $PYTHON_PACKAGES_MISSING -eq 1 ]; then
        install_python_packages
        create_env_file
        create_directories
    fi
    
    # Print final summary
    print_summary
}

# Run main function
main "$@"