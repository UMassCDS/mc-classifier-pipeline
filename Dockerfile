
# 1.  BUILD STAGE  #

FROM python:3.11-slim AS build


RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# copy and cache only the dependency list first 
COPY requirements.txt .

RUN python -m pip install --upgrade pip wheel && \
    pip wheel --wheel-dir /tmp/wheels -r requirements.txt



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

# copy just the source code needed at runtime
COPY src ./src    

# default CLI entry 
ENTRYPOINT ["python", "-m", "mc_classifier_pipeline.doc_retriever"]
CMD ["--help"]
