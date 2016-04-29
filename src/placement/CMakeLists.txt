cmake_minimum_required(VERSION 2.8.11)

project(placement)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  to_gray.cu
  summed_area_table.cu
  distance_transform.cu
  apollonius.cu
  occlusion.cu
  occlusion_calculator.cpp
  integral_costs_calculator.cu
  saliency.cu
  cost_function_calculator.cu
  labeller.cpp
  cuda_texture_mapper.cpp
  constraint_updater.cpp
  persistent_constraint_updater.cpp
  labels_arranger.h
  apollonius_labels_arranger.cpp
  )

cuda_add_library(placement-lib ${SOURCES})
target_link_libraries(placement-lib ${CUDA_LIBRARIES} ${Clipper_LIBRARIES})
