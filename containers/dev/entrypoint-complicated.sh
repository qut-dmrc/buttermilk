#!/bin/bash
set -e

# Default values
DEFAULT_USERNAME="debian"
DEFAULT_UID=1000
DEFAULT_GID=1000

# Get runtime user configuration
RUNTIME_USERNAME=${RUNTIME_USERNAME:-$DEFAULT_USERNAME}
RUNTIME_UID=${RUNTIME_UID:-$DEFAULT_UID}
RUNTIME_GID=${RUNTIME_GID:-$DEFAULT_GID}

# Check if user already exists
if ! id "$RUNTIME_USERNAME" &>/dev/null; then
    echo "Creating user: $RUNTIME_USERNAME (UID: $RUNTIME_UID, GID: $RUNTIME_GID)"

    # Create group if it doesn't exist
    if ! getent group "$RUNTIME_GID" &>/dev/null; then
        groupadd --gid "$RUNTIME_GID" "$RUNTIME_USERNAME"
    fi

    # Create user
    useradd --uid "$RUNTIME_UID" --gid "$RUNTIME_GID" -m -s /bin/bash "$RUNTIME_USERNAME"

    # Add to sudo group if not already
    if command -v sudo &>/dev/null; then
        echo "$RUNTIME_USERNAME ALL=NOPASSWD: ALL" | tee -a /etc/sudoers
    fi

    # Set up home directory permissions
    chown -R "$RUNTIME_UID:$RUNTIME_GID" "/home/$RUNTIME_USERNAME"
else
    echo "User $RUNTIME_USERNAME already exists"
fi

USER_HOME="/home/$RUNTIME_USERNAME"

# Set up SSH directory and agent permissions
sudo -u "$RUNTIME_USERNAME" mkdir -p "$USER_HOME/.ssh"
sudo -u "$RUNTIME_USERNAME" chmod 700 "$USER_HOME/.ssh"

# Fix SSH agent socket permissions if it exists
if [ -S "/ssh-agent" ]; then
    chown "$RUNTIME_USERNAME:$RUNTIME_GID" /ssh-agent
fi

# Set up git configuration
if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    echo "Configuring git..."
    sudo -u "$RUNTIME_USERNAME" git config --global user.name "$GIT_USER_NAME"
    sudo -u "$RUNTIME_USERNAME" git config --global user.email "$GIT_USER_EMAIL"
fi

# Clone and setup dotfiles
if [ -n "$DOTFILES_REPO" ]; then
    DOTFILES_DIR="$USER_HOME/.dotfiles"
    if [ ! -d "$DOTFILES_DIR" ]; then
        echo "Cloning dotfiles from $DOTFILES_REPO..."
        sudo -u "$RUNTIME_USERNAME" git clone "$DOTFILES_REPO" "$DOTFILES_DIR"

        # Look for common setup scripts
        if [ -f "$DOTFILES_DIR/install.sh" ]; then
            echo "Running dotfiles install script..."
            cd "$DOTFILES_DIR"
            sudo -u "$RUNTIME_USERNAME" bash install.sh
        elif [ -f "$DOTFILES_DIR/setup.sh" ]; then
            echo "Running dotfiles setup script..."
            cd "$DOTFILES_DIR"
            sudo -u "$RUNTIME_USERNAME" bash setup.sh
        elif [ -f "$DOTFILES_DIR/bootstrap.sh" ]; then
            echo "Running dotfiles bootstrap script..."
            cd "$DOTFILES_DIR"
            sudo -u "$RUNTIME_USERNAME" bash bootstrap.sh
        else
            echo "No setup script found in dotfiles repo"
        fi
    else
        echo "Dotfiles already cloned, pulling latest..."
        cd "$DOTFILES_DIR"
        sudo -u "$RUNTIME_USERNAME" git pull
    fi
fi

# Test SSH agent forwarding
echo "Testing SSH agent forwarding..." >&2
if timeout 3s sh -c 'SSH_AUTH_SOCK=/ssh-agent ssh-add -l >/dev/null 2>&1'; then
    echo "✅ SSH agent forwarding is working" >&2
else
    echo "❌ SSH agent forwarding failed or timed out" >&2
fi

# Set up SSH environment for the user
echo "export SSH_AUTH_SOCK=/ssh-agent" >> /home/$RUNTIME_USERNAME/.bashrc
echo "export SSH_AUTH_SOCK=/ssh-agent" >> /home/$RUNTIME_USERNAME/.profile

# Switch to the runtime user and execute the command
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    # Interactive shell
    exec su - "$RUNTIME_USERNAME" -c "cd /home/$RUNTIME_USERNAME && exec $*"
else
    # Execute command as user
    exec su - "$RUNTIME_USERNAME" -c "cd /home/$RUNTIME_USERNAME && exec $*"
fi
