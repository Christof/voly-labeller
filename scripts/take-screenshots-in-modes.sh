#!/bin/zsh

#Terminate on error
set -e

script_directory="$(dirname "$0")"
cd $script_directory
cd ../build

./voly-labeller $1 -s="$2_default"
./voly-labeller $1 --internal-labelling -s="$2_internal"
./voly-labeller $1 --optimize-on-idle -s="$2_optimized"
./voly-labeller $1 --disable-labelling --layers=1 -s="$2_no"
./voly-labeller $1 --layers=1 --hard-constraints --apollonius -s="$2_decoret"
