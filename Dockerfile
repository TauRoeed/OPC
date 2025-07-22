FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt
RUN pip install memory-profiler
RUN pip install line_profiler

# Optional: Set up a kernel (helpful if you want to name it explicitly)
RUN python -m ipykernel install --user --name=debug-kernel --display-name "Python 3.9 (debug)"

# Expose Jupyter and debugpy ports
EXPOSE 8888 5678

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]


