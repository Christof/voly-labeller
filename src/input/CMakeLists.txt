cmake_minimum_required(VERSION 2.8.11)

project(input)

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(SOURCES
  scxml_importer.cpp
  invoke_manager.cpp
  signal_manager.cpp
  event_transition.cpp
  )

if(MSVC)
  add_library(input STATIC ${SOURCES}) 
else()
  add_library(input OBJECT ${SOURCES})
endif()