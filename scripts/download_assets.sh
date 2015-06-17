#!/bin/bash

#Terminate on error
set -e
mkdir -p $(pwd -P)/$(dirname "$0")/../assets/
cd $(pwd -P)/$(dirname "$0")/../assets/

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0ByTbZ7z8JSt-T3RzWVBYaWJMU3M' -O anchor.dae
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0ByTbZ7z8JSt-d1I2R0NKZmZGZG8' -O assets.dae

#wget 'http://osirix-viewer.com/datasets/DATA/MANIX.zip'
mkdir -p datasets
#unzip MANIX.zip -d datasets
