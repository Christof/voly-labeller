cmake_minimum_required(VERSION 2.8.11)

project(placement)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  summed_area_table.cu
  distance_transform.cu
  labeller.cpp
  cuda_texture_mapper.cpp
  )

cuda_add_library(placement-lib ${SOURCES})
