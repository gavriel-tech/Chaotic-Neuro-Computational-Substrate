# GMCS Universal Platform - Production Dockerfile
# Multi-stage build for optimized image size and security

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04

# ============================================
# Stage 1: Builder - Compile dependencies
# ============================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1-dev \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION

LABEL maintainer="Gavriel Technologies"
LABEL description="GMCS Universal Chaotic-Neuro Computational Platform"
LABEL version="0.1"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    portaudio19-dev \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Create non-root user for security
RUN groupadd -r gmcs && useradd -r -g gmcs -u 1000 gmcs

# Create app directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=gmcs:gmcs src/ ./src/
COPY --chown=gmcs:gmcs pyproject.toml ./
COPY --chown=gmcs:gmcs README.md ./

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs /app/models /app/reports /app/saved_configs /app/saved_sessions && \
    chown -R gmcs:gmcs /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORMS=gpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    GMCS_HOST=0.0.0.0 \
    GMCS_PORT=8000 \
    GMCS_LOG_LEVEL=INFO \
    GMCS_AUDIO_ENABLED=false

# Switch to non-root user
USER gmcs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "src.main", "--host", "0.0.0.0", "--port", "8000", "--no-audio"]

# ============================================
# Alternative configurations (commented out)
# ============================================

# For CPU-only deployment, use this base image instead:
# FROM ubuntu:${UBUNTU_VERSION}

# For AMD ROCm support, use:
# FROM rocm/pytorch:latest

# For Apple Silicon (build on ARM64 host):
# FROM python:${PYTHON_VERSION}-slim
