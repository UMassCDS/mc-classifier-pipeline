
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
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# install prebuilt wheels (no compiler needed) 
COPY --from=build /tmp/wheels /tmp/wheels
RUN python -m pip install --upgrade pip && \
    pip install /tmp/wheels/* && \
    rm -rf /tmp/wheels                 # reclaim layer space

# Create non-root user and give ownership of /app
RUN useradd -ms /bin/bash -u 1001 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default to doc-retriever but allow override
ENTRYPOINT ["python", "-m"]
CMD ["mc_classifier_pipeline.doc_retriever", "--help"]
