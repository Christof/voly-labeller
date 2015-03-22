#!/bin/bash

#Terminate on error
set -e

cd /home/$USERNAME/Documents/sources

curl -o gtest.zip "https://googletest.googlecode.com/files/gtest-1.7.0.zip"
unzip -qo gtest.zip
cd gtest-*
./configure
make -j
sudo cp -a include/gtest /usr/include
sudo cp -a lib/.libs/* /usr/lib/
sudo ldconfig -v | grep gtest
