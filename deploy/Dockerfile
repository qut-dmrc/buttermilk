# syntax=docker/dockerfile:1.7
ARG APP_PATH="/src/buttermilk" 
ARG VENV_PATH="${APP_PATH}/.venv"

################################
# PYTHON-BASE
# Sets up all our shared environment variables
################################
FROM python:3.11-slim-bookworm AS python-base
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_CACHE_DIR=/tmp/pip_cache

# Set the timezone to Australia/Brisbane
RUN ln -sf /usr/share/zoneinfo/Australia/Brisbane /etc/localtime && \
    echo "Australia/Brisbane" | tee /etc/timezone
    
RUN apt update && apt install -y --no-install-recommends --upgrade curl less wget gnupg apt-transport-https ca-certificates ssh git

##### INSTALL NODE.JS AND NPM #######
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y nodejs

##### INSTALL UV #######
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

################################
# BUILDER-BASE
# Used to build deps + create our virtual environment
################################
FROM python-base AS builder-base
WORKDIR /src

##### COPY REQUIREMENTS FOR LATER USE #####
# Copy only the files required to install dependencies, so we separate out the layers for docker
COPY ./deploy/requirements.txt /tmp/requirements.txt

# Download packages to /wheels directory to be used in the next stage
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels && \
    # Exclude the local editable package (-e .) from the download set, since the source
    # isn't present in this stage. We'll install it later in the final stage.
    sed -E '/^[[:space:]]*-e[[:space:]]+\.[[:space:]]*$/d' /tmp/requirements.txt > /tmp/requirements.noe.txt && \
    uv pip download -r /tmp/requirements.noe.txt -o /wheels


#############################################################
##### CREATE BASE IMAGE WITH USER, SOURCE, DEPENDENCIES #####
#############################################################
FROM python-base AS buttermilk
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_PATH
ARG VENV_PATH

RUN mkdir -p ${APP_PATH}

WORKDIR ${APP_PATH}

##### COPY VENV AND SOURCE REPOSITORIES IN FULL #######
# Copy source from build context (repository root)
COPY . ${APP_PATH}
WORKDIR ${APP_PATH}

# Copy wheels from builder-base
COPY --link --from=builder-base /wheels /wheels

# Create venv and install from wheels
RUN uv venv ${VENV_PATH} --seed && \
    . ${VENV_PATH}/bin/activate && \
    uv pip sync ./deploy/requirements.txt --find-links /wheels --no-index

# Clean up wheels
RUN rm -rf /wheels

# prepend venv to path
ENV PATH="$VENV_PATH/bin:$PATH"


CMD ["bash", "-c", "${APP_PATH}/deploy/startapi.sh", "${FLOW_PARAMS}"]