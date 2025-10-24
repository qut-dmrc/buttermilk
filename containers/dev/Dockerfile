
#############################################################
#####            Buttermilk Dev Container               #####
#############################################################
FROM debian:testing AS bmdevbase

ARG TS_AUTHKEY
ARG TS_EXTRA_ARGS
ARG USERNAME=debian
ARG DEBIAN_FRONTEND=noninteractive

# Install extra packages
COPY ./containers/dev/devpackages.txt /tmp/packages.txt
RUN apt update && xargs -a /tmp/packages.txt apt install -y --no-install-recommends && apt-get autoremove -y && apt-get -y clean

##### INSTALL NODE.JS #######
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y nodejs

# Create user
RUN groupadd --gid 1000 ${USERNAME} && \
    useradd --uid 1000 --gid 1000 -m -s /bin/zsh ${USERNAME} && \
    echo "${USERNAME} ALL=NOPASSWD: ALL" >> /etc/sudoers

# Install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# Install gh cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install gh -y

# azure CLI tools
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# install docker CLI tools
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/

# Install tailscale
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list && sudo apt-get update && sudo apt-get install -y tailscale 

# install Edge
RUN curl -fSsL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/microsoft-edge.gpg > /dev/null && echo 'deb [signed-by=/usr/share/keyrings/microsoft-edge.gpg] https://packages.microsoft.com/repos/edge stable main' | sudo tee /etc/apt/sources.list.d/microsoft-edge.list && sudo apt-get update && sudo apt-get install -y --no-install-recommends microsoft-edge-beta

# Generate locales 
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Copy entrypoint script
USER root
COPY ./containers/dev/entrypoint-simple.sh /entrypoint.sh
COPY ./containers/dev/entrypoint-complicated.sh /entrypoint-complicated.sh
RUN chmod +x /entrypoint.sh /entrypoint-complicated.sh


############################ CHANGE TO USER  ############################
FROM bmdevbase AS bmdev
ARG TS_AUTHKEY
ARG TS_EXTRA_ARGS
ARG USERNAME=debian
ARG DEBIAN_FRONTEND=noninteractive

USER ${USERNAME}
WORKDIR /home/${USERNAME}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Node.js tools for user
ENV NPM_CONFIG_PREFIX=/home/${USERNAME}/.npm-global
ENV PATH=${NPM_CONFIG_PREFIX}/bin:${PATH}
RUN mkdir -p ${NPM_CONFIG_PREFIX}
RUN npm config set prefix ${NPM_CONFIG_PREFIX}

# install npm-based tools
RUN npm install -g \
    diff-so-fancy \
    @google/gemini-cli \
    @anthropic-ai/claude-code \
    wrangler

# Copy source files with proper ownership from base image
COPY --chown=${USERNAME}:${USERNAME} --from=us-central1-docker.pkg.dev/prosocial-443205/reg/buttermilk:dev /src /src

ENTRYPOINT ["/entrypoint.sh"]
CMD ["zsh"]
