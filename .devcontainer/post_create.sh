#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

USER=$(whoami)
echo "This setup script is running as $USER in `pwd`"

# create a default .zshrc file
touch /home/$USER/.zshrc || echo "Unable to create /home/$USER/.zshrc"
 
# increase the number of files we can watch at once
#echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
#sudo sysctl -p  
#echo "203.101.231.253 reg.digitalobservatory.net" | sudo tee -a /etc/hosts
