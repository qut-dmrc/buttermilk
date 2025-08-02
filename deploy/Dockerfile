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

##### INSTALL PYTHON DEPENDENCIES #####
# Copy only the files required to install dependencies, so we separate out the layers for docker
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt


################################
# BUILDER-BASE
# Used to build deps + create our virtual environment
################################
FROM python-base AS builder-base
ARG VENV_PATH
WORKDIR /src

# install runtime deps into $VENV_PATH
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python3 -m venv ${VENV_PATH} && \
    . ${VENV_PATH}/bin/activate && \
    pip install -r /tmp/requirements.txt


#############################################################
##### CREATE BASE IMAGE WITH USER, SOURCE, DEPENDENCIES #####
#############################################################
FROM python-base AS buttermilk
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_PATH
ARG VENV_PATH

RUN mkdir -p ${APP_PATH} ${VENV_PATH} 

WORKDIR ${APP_PATH}

##### COPY VENV AND SOURCE REPOSITORIES IN FULL #######
# Copy source from github actions
COPY . ${APP_PATH}
WORKDIR ${APP_PATH}

# Install virtual environment
COPY --link --from=builder-base ${VENV_PATH} ${VENV_PATH}

# prepend venv to path
ENV PATH="$VENV_PATH/bin:$PATH"

CMD ["bash", "-c", "${APP_PATH}/deploy/startapi.sh", "${FLOW_PARAMS}"]