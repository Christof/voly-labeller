cmake_minimum_required(VERSION 2.8.11)

project(voly-labeller)


find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_executable(voly-labeller src/main.cpp)

target_link_libraries(voly-labeller
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
)
