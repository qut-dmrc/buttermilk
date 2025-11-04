#!/bin/bash

# Buttermilk Container Startup Script
# Handles GCS configuration mounting and application startup

set -euo pipefail

echo "=== Buttermilk Container Startup ==="

# Environment variables with defaults
GCS_CONFIG_BUCKET="${GCS_CONFIG_BUCKET:-}"
GCS_CONFIG_PATH="${GCS_CONFIG_PATH:-buttermilk-configs}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/conf}"

# Function to check if directory is mounted
is_mounted() {
    local dir="$1"
    mount | grep -q " on $dir type fuse.gcsfuse"
}

# Function to mount GCS bucket
mount_gcs_config() {
    echo "Mounting GCS bucket '$GCS_CONFIG_BUCKET' to '$MOUNT_POINT'..."

    # Create mount point if it doesn't exist
    mkdir -p "$MOUNT_POINT"

    # Mount with gcsfuse
    gcsfuse \
        --implicit-dirs \
        --only-dir="$GCS_CONFIG_PATH" \
        --uid=$(id -u) \
        --gid=$(id -g) \
        --file-mode=644 \
        --dir-mode=755 \
        "$GCS_CONFIG_BUCKET" "$MOUNT_POINT"

    echo "GCS bucket mounted successfully"

    # Verify mount has configuration files
    if [ -f "$MOUNT_POINT/config.yaml" ]; then
        echo "Configuration files found in mounted GCS bucket"
    else
        echo "WARNING: No config.yaml found in mounted GCS bucket"
        echo "Available files:"
        ls -la "$MOUNT_POINT" || echo "Could not list mount directory"
    fi
}

# Main configuration setup
setup_configuration() {
    if [ -n "$GCS_CONFIG_BUCKET" ]; then
        echo "GCS configuration mode enabled"
        echo "Bucket: $GCS_CONFIG_BUCKET"
        echo "Path: $GCS_CONFIG_PATH"

        # Check if already mounted
        if is_mounted "$MOUNT_POINT"; then
            echo "GCS bucket already mounted at $MOUNT_POINT"
        else
            # Attempt to mount GCS bucket
            if mount_gcs_config; then
                echo "GCS configuration mounted successfully"
            else
                echo "ERROR: Failed to mount GCS bucket"
                echo "Falling back to local configuration"
                restore_local_config
            fi
        fi
    else
        echo "Using local configuration (GCS_CONFIG_BUCKET not set)"
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    echo "Shutting down container..."

    # Unmount GCS if mounted
    if [ -n "$GCS_CONFIG_BUCKET" ] && is_mounted "$MOUNT_POINT"; then
        echo "Unmounting GCS bucket..."
        fusermount -u "$MOUNT_POINT" || echo "Warning: Could not unmount GCS bucket"
    fi

    echo "Cleanup complete"
}

# Set up signal handlers for graceful shutdown
trap cleanup EXIT INT TERM

# Main execution
main() {
    setup_configuration
}

# Execute main function with all arguments
main "$@"
