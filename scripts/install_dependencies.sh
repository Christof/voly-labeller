#!/bin/bash

#Terminate on error
set -e

USER_NAME=$USERNAME
USER_HOME=/home/$USER_NAME
USER_SOURCES=$USER_HOME/Documents/sources

cd $USER_HOME

mkdir -p USER_SOURCES

sudo apt-get install libassimp-dev doxygen graphviz -y
