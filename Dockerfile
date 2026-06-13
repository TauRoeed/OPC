# OPC Python environment only. Mount your repo at run time.
#
# Build:
#   docker build -t opc .
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t opc:gpu .
#
# Run (interactive shell, repo mounted):
#   docker run -it --rm \
#     -v "$PWD:/app" \
#     -w /app \
#     --entrypoint bash \
#     opc
#
# Then inside:
#   python -m BPR.smoke_test_loaders
#   python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m

FROM python:3.12-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    libhdf5-103-1 \
    && rm -rf /var/lib/apt/lists/*

# Only dependency spec at build time — application code is mounted at run time.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url https://pypi.org/simple \
    && rm /tmp/requirements.txt

ENTRYPOINT ["python"]
CMD ["--version"]
