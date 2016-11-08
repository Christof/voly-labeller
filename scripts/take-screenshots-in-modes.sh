#!/bin/zsh

#Terminate on error
set -e

script_directory="$(dirname "$0")"
cd $script_directory
cd ../build


function getWithSuffix {
  ALL=""
  for i in $(echo $1 | tr "," "\n")
  do
    ALL+="$i"
    ALL+="_"
    ALL+="$2"
    ALL+=","
  done

  PARAMS=${ALL:0:${#ALL}-1}
}

getWithSuffix $2 "default"
./voly-labeller $1 -s=$PARAMS

getWithSuffix $2 "internal"
./voly-labeller $1 --internal-labelling -s="$PARAMS"

getWithSuffix $2 "optimized"
./voly-labeller $1 --optimize-on-idle -s="$PARAMS"

getWithSuffix $2 "no"
./voly-labeller $1 --disable-labelling --layers=1 -s="$PARAMS"

getWithSuffix $2 "decoret"
./voly-labeller $1 --layers=1 --hard-constraints --apollonius -s="$PARAMS"
