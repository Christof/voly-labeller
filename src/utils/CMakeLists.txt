cmake_minimum_required(VERSION 2.8.11)

project(utils)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  persister.cpp
  )

add_library(utils OBJECT ${SOURCES})
