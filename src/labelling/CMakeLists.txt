cmake_minimum_required(VERSION 2.8.11)

project(labelling)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  labels.cpp
  clustering.cpp
  )

add_library(labelling OBJECT ${SOURCES})
