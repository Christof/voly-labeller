#!/bin/bash

#Terminate on error
set -e

sudo apt-get install libassimp-dev doxygen graphviz freeglut3-dev libeigen3-dev libboost-all-dev libmagick++-dev libfftw3-dev \
  libinsighttoolkit4-dev libgdcm-tools vtk-dicom-tools python-vtkgdcm \
  libavcodec-dev libswscale-dev libavformat-dev -y

