#!/bin/bash

# ==============================================================================
# Script: iamcopy.sh
# Description: This script copies all permissions from a source IAM role to a
#              destination IAM role using the gcloud command-line tool.
# Usage: ./iamcopy.sh <source_role_id> <destination_role_id>
# Example: ./iamcopy.sh roles/container.admin my_new_role
# ==============================================================================

# --- Argument Validation ------------------------------------------------------
# Check if exactly two arguments were provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_role_id> <destination_role_id>"
    echo "Example: $0 roles/container.admin my_new_role"
    exit 1
fi

SOURCE_ROLE="$1"
DEST_ROLE="$2"

echo "Copying permissions from '$SOURCE_ROLE' to '$DEST_ROLE'..."
echo "------------------------------------------------------------------"

# --- Get Permissions from Source Role -----------------------------------------
# The gcloud command describes the source role and formats the output to
# only include the 'includedPermissions' field.
# The `value` format outputs a space-separated string of permissions.
# `tr ' ' ','` is then used to replace spaces with commas, as required by the
# `--add-permissions` flag for the update command.
PERMISSIONS=$(gcloud iam roles describe "$SOURCE_ROLE" --format="value(includedPermissions)")
if [ $? -ne 0 ]; then
    echo "Error: Failed to describe the source role '$SOURCE_ROLE'."
    exit 1
fi

PERMISSIONS_CSV=$(echo "$PERMISSIONS" | tr ' ' ',')
if [ -z "$PERMISSIONS_CSV" ]; then
    echo "Warning: The source role '$SOURCE_ROLE' has no permissions to copy."
    exit 0
fi

# --- Grant Permissions to Destination Role ------------------------------------
# The gcloud command updates the destination role, adding all the permissions
# from the comma-separated list we just created.
echo "Adding permissions to the destination role..."
gcloud iam roles update "$DEST_ROLE" --add-permissions="$PERMISSIONS_CSV"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------------------------"
    echo "Successfully copied permissions from '$SOURCE_ROLE' to '$DEST_ROLE'."
else
    echo "Error: Failed to update the destination role '$DEST_ROLE'."
    exit 1
fi
