
# 1.  BUILD STAGE  #

FROM python:3.11-slim AS build


RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# copy only the dependency list first 
COPY pyproject.toml ./

COPY src ./src

#Build wheels for our package + all its deps
RUN python -m pip install --upgrade pip wheel && \
    pip wheel --wheel-dir /tmp/wheels .



# 2.  RUNTIME STAGE #

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src             
WORKDIR /app

# install prebuilt wheels (no compiler needed) 
COPY --from=build /tmp/wheels /tmp/wheels
RUN python -m pip install --upgrade pip && \
    pip install /tmp/wheels/* && \
    rm -rf /tmp/wheels                 # reclaim layer space

# copy source code for runtime execution
COPY src ./src    

# Create a flexible entry point script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'case "$1" in' >> /app/entrypoint.sh && \
    echo '  "run-pipeline") shift; exec python -m mc_classifier_pipeline.run_pipeline "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "model-orchestrator") shift; exec python -m mc_classifier_pipeline.model_orchestrator "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "inference") shift; exec python -m mc_classifier_pipeline.inference "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "preprocess") shift; exec python -m mc_classifier_pipeline.preprocessing "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "train") shift; exec python -m mc_classifier_pipeline.trainer "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "evaluate") shift; exec python -m mc_classifier_pipeline.evaluation "$@" ;;' >> /app/entrypoint.sh && \
    echo '  "doc-retriever") shift; exec python -m mc_classifier_pipeline.doc_retriever "$@" ;;' >> /app/entrypoint.sh && \
    echo '  *) echo "Usage: $0 {run-pipeline|model-orchestrator|inference|preprocess|train|evaluate|doc-retriever} [args...]"; exit 1 ;;' >> /app/entrypoint.sh && \
    echo 'esac' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Flexible entry point that accepts different pipeline actions
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["doc-retriever", "--help"]
