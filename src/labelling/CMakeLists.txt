cmake_minimum_required(VERSION 2.8.11)

project(labelling)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  labels_container.cpp
  label.cpp
  labels.cpp
  label_positions.cpp
  clustering.cpp
  )

if(MSVC)
  add_library(labelling STATIC ${SOURCES}) 
else()
  add_library(labelling OBJECT ${SOURCES})
endif()
