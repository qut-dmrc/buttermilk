#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

USER=$(whoami)
echo "This setup script is running as $USER in `pwd`"

# create a default .zshrc file
touch /home/$USER/.zshrc || echo "Unable to create /home/$USER/.zshrc"
 
# increase the number of files we can watch at once
#echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
#sudo sysctl -p  

# Consider using a different directory for global npm packages
# # Create a directory for your global packages
# mkdir -p ~/.npm-global

# # Configure npm to use the new directory path
# npm config set prefix ~/.npm-global

# # Note: Replace ~/.bashrc with ~/.zshrc, ~/.profile, or other appropriate file for your shell
# echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc

# # Apply the new PATH setting
# source ~/.bashrc

# # Now reinstall Claude Code in the new location
# npm install -g @anthropic-ai/claude-code
echo 'alias claude="/home/debian/.claude/local/claude"' >> /home/$USER/.zshrc

# Optional: Reinstall your previous global packages in the new location
# Look at ~/npm-global-packages.txt and install packages you want to keep
# npm install -g package1 package2 package3...
# Optional: Reinstall your previous global packages in the new location
# Look at ~/npm-global-packages.txt and install packages you want to keep
# npm install -g package1 package2 package3...
