cmake_minimum_required(VERSION 2.8.11)

project(placement)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  to_gray.cu
  summed_area_table.cu
  distance_transform.cu
  apollonius.cu
  occupancy.cu
  cost_function_calculator.cu
  occupancy_updater.cu
  labeller.cpp
  cuda_texture_mapper.cpp
  constraint_updater.cpp
  )

cuda_add_library(placement-lib ${SOURCES})
target_link_libraries(placement-lib ${CUDA_LIBRARIES})
