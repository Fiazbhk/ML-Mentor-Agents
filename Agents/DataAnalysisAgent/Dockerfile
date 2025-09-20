# Multi-stage build for optimal image size
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create and set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -e .

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create non-root user for security
RUN groupadd -r mlmentor && useradd -r -g mlmentor mlmentor

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/temp && \
    chown -R mlmentor:mlmentor /app

# Switch to non-root user
USER mlmentor

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="ML Mentor Team <team@mlmentor.dev>" \
      version="1.0.0" \
      description="DataAnalysisAgent for intelligent dataset analysis" \
      org.coral.agent.name="DataAnalysisAgent" \
      org.coral.agent.version="1.0.0" \
      org.coral.agent.category="data-processing"