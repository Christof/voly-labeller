cmake_minimum_required(VERSION 2.8.11)

project(forces)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  label_state.cpp
  force.cpp
  center_force.cpp
  anchor_force.cpp
  label_collision_force.cpp
  lines_crossing_force.cpp
  labeller.cpp
  )

add_library(forces OBJECT ${SOURCES})
