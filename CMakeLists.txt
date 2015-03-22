cmake_minimum_required(VERSION 2.8.11)

project(voly-labeller)

option(test "Build all tests." ON)
if(test)
  enable_testing()
  add_subdirectory(test)
endif(test)


find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Quick REQUIRED)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
  message(ERROR "OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(PkgConfig)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
#find_package(Eigen3 REQUIRED)

pkg_check_modules(ASSIMP REQUIRED assimp)
include_directories(${ASSIMP_INCLUDE_DIRS})

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${Qt5Quick_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES src/resources.qrc)

set(SOURCES
  src/window.cpp
  src/camera.cpp
  src/shader_program.cpp
  src/mesh.cpp
  src/main.cpp
  src/abstract_scene.h
  src/demo_scene.cpp
  ${RESOURCES})

add_executable(voly-labeller ${SOURCES})

target_link_libraries(voly-labeller
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Quick
  ${OPENGL_LIBRARIES}
  ${ASSIMP_LIBRARIES}
)

add_custom_target(valgrind valgrind -v --leak-check=full --show-leak-kinds=all --tool=memcheck --suppressions=../config/valgrind_suppressions.txt ./voly-labeller)
add_custom_target(helgrind valgrind -v --tool=helgrind ./voly-labeller)
# Analyzers
add_custom_target(lint python ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpplint.py --root=${CMAKE_CURRENT_SOURCE_DIR} --filter=-whitespace/braces,-legal/copyright,-readability/streams,-runtime/references,-runtime/int,-whitespace/indent ${CMAKE_CURRENT_SOURCE_DIR}/src/*.* ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.*)
add_custom_target(cppcheck cppcheck ${CMAKE_CURRENT_SOURCE_DIR}/src --enable=all VERBATIM)

# Documentation
add_custom_target(doc cd ${CMAKE_CURRENT_SOURCE_DIR} && doxygen)
add_custom_target(doc-open xdg-open ${CMAKE_CURRENT_SOURCE_DIR}/doc/html/index.html)

