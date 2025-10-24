#!/bin/bash
set -e

# Simple entrypoint - just ensure SSH agent socket has right permissions
if [ -S "/ssh-agent" ]; then
    chmod 666 /ssh-agent
fi

# Execute the command
exec "$@"
