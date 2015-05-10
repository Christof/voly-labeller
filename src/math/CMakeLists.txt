cmake_minimum_required(VERSION 2.8.11)

project(math)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  aabb2d.cpp
  obb.cpp
  )

add_library(math OBJECT ${SOURCES})
