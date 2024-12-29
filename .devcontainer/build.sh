#!/bin/bash

docker pull us-central1-docker.pkg.dev/prosocial-443205/reg/prosocialdev:latest
devcontainer up --workspace-folder . --config .devcontainer/devcontainer.json --remove-existing-container
