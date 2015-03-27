cmake_minimum_required(VERSION 2.8.11)

project(tests)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ../src/resources.qrc)

set(SOURCES
  ../src/window.cpp
  ../src/camera.cpp
  ../src/shader_program.cpp
  ../src/mesh.cpp
  ../src/abstract_scene.h
  ../src/demo_scene.cpp
  ../src/input/scxml_importer.cpp
  ../src/input/invoke_manager.cpp
  ${RESOURCES})

message(sources: ${SOURCES})
file(GLOB_RECURSE TESTS ${CMAKE_CURRENT_SOURCE_DIR}**/*.cc)
add_executable(tests ${TESTS} ${SOURCES})
target_link_libraries(tests
  ${GTEST_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Quick
  ${OPENGL_LIBRARIES}
  ${ASSIMP_LIBRARIES}
  )

add_test(tests tests)
