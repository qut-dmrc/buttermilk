# syntax=docker/dockerfile:1.7
ARG APP_PATH="/src/buttermilk" 
ARG VENV_PATH="${APP_PATH}/.venv"

################################
# PYTHON-BASE
################################
FROM python:3.11-slim-bookworm AS python-base
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_CACHE_DIR=/tmp/pip_cache

# Set the timezone to Australia/Brisbane
RUN ln -sf /usr/share/zoneinfo/Australia/Brisbane /etc/localtime && \
    echo "Australia/Brisbane" | tee /etc/timezone

# Base runtime deps only  
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl less wget gnupg apt-transport-https ca-certificates ssh git \
    && rm -rf /var/lib/apt/lists/*

##### INSTALL NODE.JS AND NPM #######
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

##### INSTALL UV #######
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

################################
# BUILDER-BASE
################################
FROM python-base AS builder-base
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage layer caching
COPY ./deploy/requirements.txt /tmp/requirements.txt

# Exclude editable -e . 
RUN sed -E '/^[[:space:]]*-e[[:space:]]+\.[[:space:]]*$/d' /tmp/requirements.txt > /tmp/requirements.txt

# Prefetch/build wheels to speed up final install
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    python -m pip wheel -r /tmp/requirements.txt -w /wheels

################################
# FINAL
################################
FROM python-base AS buttermilk
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV UV_LINK_MODE=copy
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_PATH
ARG VENV_PATH

RUN mkdir -p ${APP_PATH}
WORKDIR ${APP_PATH}

# Copy source
COPY . ${APP_PATH}

# Copy wheelhouse
COPY --link --from=builder-base /wheels /wheels

# Exclude editable -e . 
RUN sed -E '/^[[:space:]]*-e[[:space:]]+\.[[:space:]]*$/d' /tmp/requirements.txt > /tmp/requirements.txt

# Create venv and install; prefer wheels, allow network fallback
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv ${VENV_PATH} --seed && \
    . ${VENV_PATH}/bin/activate && \
    uv pip sync /tmp/requirements.txt --find-links /wheels

# Clean up wheelhouse
RUN rm -rf /wheels

# Prepend venv to PATH
ENV PATH="$VENV_PATH/bin:$PATH"

CMD ["bash", "-c", "echo 'container ready; sleeping...'; sleep infinity"]