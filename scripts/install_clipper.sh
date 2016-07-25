#!/bin/bash

#Terminate on error
set -e

mkdir clipper-install-temp
cd clipper-install-temp

wget "http://downloads.sourceforge.net/project/polyclipping/clipper_ver6.2.1.zip?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fpolyclipping%2Ffiles%2F&ts=1469468623" -O clipper.zip
unzip clipper.zip
cd cpp
sed -i -- 's/INSTALL (TARGETS polyclipping LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")/INSTALL (TARGETS polyclipping LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")/g' CMakeLists.txt
cmake -DBUILD_SHARED_LIBS=OFF .

make -j2
sudo make install

cd ..
rm -rf clipper-install-temp
