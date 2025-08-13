# 1. BUILD STAGE
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only the dependency list first 
COPY pyproject.toml ./

# Install deps (no cache, binary wheels if possible)
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary .

# Copy source code
COPY src ./src


# 2. RUNTIME STAGE
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add src to PYTHONPATH so mc_classifier_pipeline is importable
ENV PYTHONPATH="/app/src"

WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /usr/local /usr/local
COPY src ./src

# # Create non-root user
# RUN useradd -ms /bin/bash -u 1001 appuser && \
#     chown -R appuser:appuser /app
# USER appuser

CMD ["mc_classifier", "doc_retriever", "--help"]
