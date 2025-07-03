# Use an official Python 3.9 image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter (optional: include notebook/lab as needed)
RUN pip install --upgrade pip && pip install \
    notebook \
    jupyterlab

# Copy requirements and install Python packages
COPY req.txt .
RUN python3.9 -m pip install -r req.txt
# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
